"""
Trains a character-level language model.

--- 学习指南 (Learning Guide) ---
这是最经典的 GPT 入门任务：Char-level Language Modeling。
任务：给定一段文本，预测下一个字符。
数据集：input.txt (通常是一本小说，或者莎士比亚全集)

数据处理：
如果 input.txt 是 "hello world"，block_size=3
训练样本可能长这样：
x: "hel" -> y: "ell" (即输入"h"预测"e", 输入"he"预测"l", 输入"hel"预测"l")
x: "ell" -> y: "llo"
...
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():
    
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        # 构建词表：找出所有唯一的字符
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        # 建立字符到整数的映射 (stoi) 和 整数到字符的映射 (itos)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # 取出长度为 block_size + 1 的一段文本
        # 为什么要 +1？因为我们要用前 block_size 个预测后 block_size 个
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long) # 输入: 0 ~ T-1
        y = torch.tensor(dix[1:], dtype=torch.long)  # 目标: 1 ~ T
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    # 读取当前目录下的 input.txt 文件
    if not os.path.exists('input.txt'):
        # 如果文件不存在，创建一个 dummy 文件用于演示
        print("input.txt not found, creating a dummy file for demonstration...")
        with open('input.txt', 'w') as f:
            f.write("Hello world! This is a test file for minGPT. " * 100)
            
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                # 设定一个初始文本 (Context)
                context = "O God, O God!"
                # 将 Context 编码为整数 Tensor
                x = torch.tensor([train_dataset.stoi.get(s, 0) for s in context], dtype=torch.long)[None,...].to(trainer.device)
                # 调用 model.generate 生成 500 个新字符
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                # 解码生成的整数序列回文本
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
