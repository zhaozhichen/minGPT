"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

--- 学习指南 (Learning Guide) ---
本文件定义了一个通用的 `Trainer` 类，用于管理训练过程。
虽然它是为 GPT 设计的，但其核心逻辑（Forward -> Backward -> Update）适用于绝大多数 PyTorch 模型。

核心流程：
1.  配置 (Config): 设定 Batch Size, Learning Rate, Max Iters 等。
2.  模型 (Model): 传入我们在 model.py 中定义的 GPT 模型。
3.  数据集 (Dataset): 提供训练数据。
4.  优化器 (Optimizer): 自动从模型中获取（区分权重衰减）。
5.  训练循环 (Run Loop):
    - 从 DataLoader 取一个 Batch (x, y)
    - 前向传播 (Forward): 计算 logits 和 loss
    - 反向传播 (Backward): 计算梯度
    - 参数更新 (Step): 更新权重
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        # 自动选择设备，通常是 'cuda' (GPU) 或 'cpu'
        C.device = 'auto'
        # dataloder parameters
        # 数据加载的工作线程数
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None  # 最大训练迭代次数
        C.batch_size = 64   # 批大小：一次训练多少个样本
        C.learning_rate = 3e-4 # 学习率：参数更新的步长
        C.betas = (0.9, 0.95) # AdamW 优化器的 beta 参数
        C.weight_decay = 0.1 # 权重衰减：用于正则化，防止过拟合
        C.grad_norm_clip = 1.0 # 梯度裁剪：防止梯度爆炸
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        
        # 将模型移动到指定设备 (GPU/CPU)
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        """
        添加回调函数。
        例如：在每个 batch 结束时打印 loss，或者定期保存模型。
        """
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        # 调用模型内部的 configure_optimizers 方法来创建优化器
        # 它可以区分哪些参数需要权重衰减 (weight decay)，哪些不需要 (如 bias, layernorm)
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        # 创建 PyTorch DataLoader
        # 它负责打乱数据 (shuffle)，分批 (batch)，并多线程加载 (num_workers)
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True, # 锁页内存，加速从 CPU 到 GPU 的传输
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train() # 设置模型为训练模式 (启用 Dropout, BatchNorm 等)
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        
        # 开始无限循环，直到达到 max_iters
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                # 如果数据用完了，重新创建一个迭代器 (虽然上面的 RandomSampler 设了很大 num_samples，通常不应该发生)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # 将数据移动到 GPU (或 CPU)
            # batch 包含 x (输入) 和 y (目标)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            # 前向传播：计算预测结果 logits 和损失 loss
            # x: (Batch, Time) -> logits: (Batch, Time, Vocab), loss: scalar
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            # 1. 清空梯度 (设为 None 比 0 更高效)
            model.zero_grad(set_to_none=True)
            # 2. 反向传播：计算 loss 对每个参数的梯度
            self.loss.backward()
            # 3. 梯度裁剪：防止梯度过大破坏模型权重
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            # 4. 更新参数：theta = theta - learning_rate * gradient
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
