"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

--- 学习指南 (Learning Guide) ---
为了帮助理解，我们将使用一个贯穿全文的【运行示例 (Running Example)】：
假设我们正在处理一个极简的场景：
- Batch Size (B) = 1  (一次处理一句话)
- Sequence Length (T) = 5  (这句话有5个token，例如 ["我", "爱", "你", "中", "国"])
- Embedding Dimension (C, n_embd) = 48  (每个token用一个48维向量表示)
- Number of Heads (n_head) = 3  (注意力机制分为3个头)
- Head Size (hs) = C / n_head = 48 / 3 = 16 (每个头处理16维)
- Vocab Size = 100 (词表里总共只有100个词)

在代码注释中，你会看到类似 shape: (B, T, C) -> (1, 5, 48) 的追踪说明。
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    
    原理：
    GELU (Gaussian Error Linear Unit) 是一种激活函数。
    相比于 ReLU (x if x>0 else 0) 在 0 点的生硬转折，GELU 提供了一个平滑的非线性变换。
    它近似于 x * P(X <= x)，其中 X 是标准正态分布。
    简单理解：它也是保留正值，抑制负值，但在 0 附近有更平滑的过渡，有助于深层模型的训练。
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    这是 Transformer 中最核心的组件：多头因果自注意力机制 (Multi-Head Causal Self-Attention)。
    
    "因果 (Causal)": 意味着在预测第 t 个词时，模型只能看到 t 及其之前的词，不能看到未来的词 (t+1, ...)。
    "自注意力 (Self-Attention)": 序列中的每个词都要去关注序列中的其他词，计算相关性。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # 我们需要为每个 token 计算 Query(查询), Key(键), Value(值) 三个向量。
        # 这里使用一个大的 Linear 层一次性计算出 3 * n_embd 维度的向量，然后切分。
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        # 注意力计算完成后的输出投影层，用于融合各个头的信息
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # 创建一个下三角矩阵 (tril)，用于 mask 掉未来的信息。
        # shape: (1, 1, block_size, block_size)。block_size 是模型能处理的最大长度。
        # 也就是 attention 矩阵中，上三角部分（未来）将被设为 -inf。
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # x 是输入张量
        # Running Example:
        # x shape: (B, T, C) = (1, 5, 48)
        # 代表 1 个句子，长度为 5，每个词向量 48 维。
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        # 1. 线性投影: 将 x 映射到 3倍维度
        # self.c_attn(x) shape: (1, 5, 48) -> (1, 5, 144)  (144 = 48 * 3)
        
        # 2. 切分 (split): 将 144 拆分为 q, k, v
        # q, k, v shape: (1, 5, 48) each
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 3. 变换形状 (view + transpose) 以适应多头注意力
        # 我们希望把 n_head (3) 作为一个独立的维度，以便并行计算。
        # C (48) 被拆分为 n_head (3) * head_size (16)。
        # .view: (1, 5, 48) -> (1, 5, 3, 16)  [B, T, n_head, hs]
        # .transpose(1, 2): 交换 T 和 n_head 维度 -> (1, 3, 5, 16) [B, n_head, T, hs]
        # 这样做的目的是让 (B, n_head) 共同作为 batch 维度，对每个头内部进行矩阵乘法。
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) -> (1, 3, 5, 16)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) -> (1, 3, 5, 16)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) -> (1, 3, 5, 16)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        
        # 4. 计算注意力分数 (Attention Scores)
        # 公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        # Matrix Multiplication (矩阵乘法): Q @ K^T
        # q: (1, 3, 5, 16)
        # k.transpose(-2, -1): (1, 3, 16, 5)  <- 转置最后两个维度，即转置 T 和 hs
        # 乘法结果 att shape: (1, 3, 5, 5) [B, n_head, T, T]
        # 这个 (T, T) 矩阵 (5x5) 里的每一个值 (i, j) 代表第 i 个 token 对第 j 个 token 的关注度。
        
        # Scaling (缩放): * (1.0 / math.sqrt(k.size(-1)))
        # 除以 sqrt(head_size) 即 sqrt(16)=4。
        # 作用：防止点积结果过大导致 softmax 进入梯度极小的饱和区 (vanishing gradients)。
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 5. 因果遮蔽 (Causal Masking)
        # 我们不能让第 1 个词看到第 2, 3, 4, 5 个词。
        # self.bias 是一个下三角全 1 矩阵 (tril)。
        # masked_fill: 将 bias 为 0 (即上三角部分，未来信息) 的位置对应的 att 值设为负无穷 (-inf)。
        # 这样 softmax 之后，这些位置的概率就会变成 0。
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # 6. 归一化 (Softmax)
        # 对最后一个维度 (T) 进行 softmax。
        # 使得每一行 (每个 token 对所有 token 的关注度) 加起来等于 1。
        att = F.softmax(att, dim=-1)
        
        # Dropout
        att = self.attn_dropout(att)
        
        # 7. 聚合值 (Aggregate Values)
        # att: (1, 3, 5, 5) [B, nh, T, T]
        # v:   (1, 3, 5, 16) [B, nh, T, hs]
        # 矩阵乘法: (5x5) @ (5x16) -> (5x16) (对每个 head 内部进行)
        # 结果 y shape: (1, 3, 5, 16) [B, nh, T, hs]
        # 这步做的是：根据关注度权重，对 Value 向量进行加权求和。
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # 8. 重组 (Re-assemble)
        # 我们要把多头拆分的结果拼回去。
        # .transpose(1, 2): (1, 3, 5, 16) -> (1, 5, 3, 16) [B, T, nh, hs] (交换 T 和 nh)
        # .contiguous(): 内存连续化，为 view 做准备
        # .view: 合并 nh 和 hs -> (1, 5, 48) [B, T, C]
        # 此时 shape 变回了和输入 x 一样。
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # 9. 输出投影
        # 经过一个 Linear 层混合各个头的信息。
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """
    """
    一个标准的 Transformer Block。
    结构：
    Input -> LN -> Attn -> Add (Residual) -> LN -> MLP -> Add (Residual) -> Output
    
    注意：minGPT 使用的是 "Pre-Norm" 结构 (GPT-2 风格)，
    即 LayerNorm 在 Attention 和 MLP 之前使用 (x = x + attn(ln(x)))。
    原始 Transformer (Attention Is All You Need) 使用的是 "Post-Norm" (x = ln(x + attn(x)))。
    Pre-Norm 训练更稳定。
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd), # 升维 4倍
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd), # 降维回原维度
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        # MLP 前向传播逻辑: Linear(升) -> GELU -> Linear(降) -> Dropout
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        # x shape: (B, T, C)
        
        # 1. Attention Sub-layer
        # LayerNorm -> Attention -> Residual Add
        # "残差连接 (Residual Connection)": x = x + ...
        # 这让梯度可以直接流过，缓解深层网络梯度消失问题。
        x = x + self.attn(self.ln_1(x))
        
        # 2. MLP Sub-layer
        # LayerNorm -> MLP -> Residual Add
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """
    """
    GPT (Generative Pre-trained Transformer) 整体模型结构。
    """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            # Token Embeddings: 将整数 token ID 转换为向量
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embeddings: 学习出来的绝对位置编码
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout
            drop = nn.Dropout(config.embd_pdrop),
            # Transformer Blocks 堆叠
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最后的 LayerNorm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # LM Head: 将最终的向量映射回词表大小，输出 logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        # idx 是输入的 token 索引序列
        # Running Example: idx = [[12, 45, 99, 18, 66]] (代表 ["我", "爱", "你", "中", "国"])
        # shape: (B, T) = (1, 5)
        
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # 1. 准备位置索引
        # pos shape: (1, 5) -> [[0, 1, 2, 3, 4]]
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        
        # 2. Token Embeddings Lookup
        # 将每个整数索引转换为 n_embd 维向量
        # idx (1, 5) -> tok_emb (1, 5, 48)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        # 3. Position Embeddings Lookup
        # 将位置索引 [0, 1, 2, 3, 4] 转换为 n_embd 维向量
        # pos (1, 5) -> pos_emb (1, 5, 48)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        # 4. Embedding 融合
        # 直接相加。这就是 Transformer 知道"词序"的唯一方式。
        # x shape: (1, 5, 48)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 5. 经过 N 层 Transformer Blocks
        # 这里 x 会依次经过每一个 Block。
        # 形状始终保持 (B, T, C) = (1, 5, 48) (因为使用了 Residual connection 和 projections)
        for block in self.transformer.h:
            x = block(x)
            
        # 6. 最终 LayerNorm
        # x shape: (1, 5, 48)
        x = self.transformer.ln_f(x)
        
        # 7. Language Model Head
        # 将 hidden state 投影回词表大小，得到 logits
        # logits shape: (1, 5, 100) (假设 vocab_size=100)
        # 每一个位置 t 的输出，代表预测第 t+1 个词的概率分布 (未归一化)。
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # 计算损失
            # 展平 logits: (B*T, vocab_size) -> (5, 100)
            # 展平 targets: (B*T) -> (5)
            # CrossEntropyLoss 会内部做 Softmax 并计算负对数似然
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        
        自回归生成 (Autoregressive Generation) 循环。
        原理：
        1. 输入当前序列 idx。
        2. 模型预测下一个词的 logits。
        3. 根据 logits 采样出下一个词 idx_next。
        4. 将 idx_next 拼接到 idx 后面。
        5. 重复上述步骤。
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            # 我们只需要最后一个时间步的预测结果 (预测下一个词)
            # logits shape: (B, T, vocab_size)
            # logits[:, -1, :] shape: (B, vocab_size)
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            # 将新生成的 token 拼接到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
