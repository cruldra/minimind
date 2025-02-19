import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    # 使用CrossEntropyLoss作为损失函数，并设置reduction='none'以保留每个样本的损失值
    # 这样做的原因：
    # 1. 我们需要对每个token计算单独的损失，而不是直接对整个batch求平均
    # 2. 后续会使用loss_mask来过滤掉padding部分的损失
    # 3. 保留每个样本的损失值可以让我们更灵活地处理不同位置的损失
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将输入数据X、目标标签Y和损失掩码loss_mask转移到指定设备（如GPU）
        # 这一步解决了以下问题：
        # 1. 确保所有张量都在同一设备上计算，避免CPU和GPU之间的数据传输开销
        # 2. 利用GPU的并行计算能力加速模型训练
        # 3. 支持分布式训练场景，确保数据在正确的设备上处理
        # 4. 为后续的模型前向传播和损失计算做好准备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 计算当前学习率，使用余弦退火策略
        # 这一步解决了以下问题：
        # 1. 动态调整学习率：根据训练进度自动调整学习率，避免固定学习率带来的训练效率低下
        # 2. 平滑过渡：使用余弦函数实现学习率的平滑变化，避免学习率突变对模型训练造成冲击
        # 3. 训练稳定性：初始阶段使用较大学习率快速收敛，后期使用较小学习率精细调整
        # 4. 自适应调度：根据当前epoch和step计算精确的学习率，实现更细粒度的控制
        # 5. 超参数解耦：将学习率计算逻辑封装在get_lr函数中，使主训练逻辑更清晰
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # 前向传播
            #    model(X) → 触发 __call__ → 执行 forward → 返回计算结果
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        # 使用GradScaler进行损失缩放和反向传播
        # 这一步解决了以下问题：
        # 1. 混合精度训练：在float16/bfloat16精度下，数值范围较小，直接计算梯度可能导致下溢
        # 2. 梯度稳定性：通过缩放损失值，确保梯度在合理范围内，避免梯度爆炸或消失
        # 3. 训练效率：在保持数值稳定性的同时，利用低精度计算加速训练过程
        # 4. 内存优化：使用低精度计算减少显存占用，允许使用更大的batch size
        # 5. 自动管理：GradScaler自动处理损失缩放和梯度反缩放，简化了混合精度训练的实现
        scaler.scale(loss).backward()

        # 当达到梯度累积步数时执行以下操作
        # 这一步解决了以下问题：
        # 1. 梯度累积：通过累积多个小batch的梯度来模拟大batch训练，解决显存不足的问题
        # 2. 梯度反缩放：在混合精度训练中，将缩放后的梯度还原为原始值，确保优化器更新参数的正确性
        # 3. 梯度裁剪：防止梯度爆炸，通过clip_grad_norm_将梯度范数限制在合理范围内，提高训练稳定性
        # 4. 参数更新：使用优化器更新模型参数，实现模型的学习和优化
        # 5. 缩放因子更新：根据梯度情况调整缩放因子，确保混合精度训练的数值稳定性
        # 6. 梯度清零：高效地清除梯度，为下一轮梯度计算做准备，set_to_none=True可以减少内存分配开销
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 更新缩放因子

            optimizer.zero_grad(set_to_none=True)  # 清除梯度

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 计算每次迭代处理的token总数，用于监控训练进度和计算吞吐量
    # 1. args.batch_size: 每个batch的样本数量
    # 2. lm_config.max_seq_len: 每个样本的最大序列长度（token数量）
    # 3. tokens_per_iter: 每次迭代处理的token总数 = batch_size * max_seq_len
    # 这个值对于：
    # - 计算训练速度（tokens/second）
    # - 估算训练时间
    # - 监控显存使用情况
    # - 调整batch size和序列长度
    # 都很有帮助
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    # 设置随机种子为1337，确保实验的可重复性
    # 1. 控制PyTorch的随机数生成器，使每次运行程序时生成的随机数相同
    # 2. 保证模型初始化、数据shuffle等随机操作的结果一致
    # 3. 便于调试和比较不同超参数设置下的模型性能
    # 4. 在多机多卡训练中，确保各进程的随机状态一致
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 创建自动混合精度(AMP)上下文管理器
    # 当device_type为"cpu"时使用nullcontext()，即不进行任何操作
    # 当device_type为"cuda"时使用torch.cuda.amp.autocast()，启用自动混合精度
    # 自动混合精度可以：
    # 1. 减少显存占用：将部分计算从float32转换为float16/bfloat16
    # 2. 提高计算速度：float16/bfloat16运算比float32更快
    # 3. 保持模型精度：只在安全的情况下进行精度转换，避免影响模型效果
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    # 检查是否在分布式数据并行(DDP)模式下运行
    # 通过检查环境变量RANK是否存在来判断
    # RANK是PyTorch DDP为每个进程分配的全局唯一标识符
    # 如果RANK存在且不为-1，说明当前在DDP模式下运行
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?

    # 初始化本地rank和设备
    # ddp_local_rank: 当前进程在本地节点中的rank，单机情况下默认为0
    # DEVICE: 默认使用第一个GPU设备"cuda:0"
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 如果是在DDP模式下运行
    if ddp:
        # 初始化分布式训练环境
        # 1. 设置进程组
        # 2. 配置通信后端（如NCCL）
        # 3. 同步各进程
        init_distributed_mode()
        # 更新args中的设备信息
        args.device = torch.device(DEVICE)

    # 初始化Wandb日志记录
    # 仅在以下情况初始化Wandb：
    # 1. 用户启用了Wandb (args.use_wandb为True)
    # 2. 不在DDP模式下运行，或者是在DDP模式下但当前进程是主进程（rank为0）
    # 这样可以避免多个进程重复记录日志
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        # 初始化Wandb项目
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        # 如果不满足上述条件，将wandb设为None
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # 创建分布式采样器（如果使用DDP）或None（单机训练）
    # 在DDP模式下，DistributedSampler确保：
    # 1. 每个进程获得不同的数据子集，避免数据重复
    # 2. 数据在多个epoch间正确打乱
    # 3. 各进程间数据分布均衡
    train_sampler = DistributedSampler(train_ds) if ddp else None
    
    # 创建数据加载器
    # 参数说明：
    # - train_ds: 训练数据集
    # - batch_size: 每个batch的大小
    # - pin_memory=True: 将数据固定到GPU内存，加速CPU到GPU的数据传输
    # - drop_last=False: 保留最后一个不完整的batch
    # - shuffle=False: 禁用默认打乱，因为DistributedSampler会处理打乱
    # - num_workers: 数据加载的线程数
    # - sampler: 使用自定义采样器（DDP模式）或None（单机模式）
    # 这样配置解决了：
    # 1. 高效的数据加载和传输
    # 2. DDP模式下的数据分布问题
    # 3. 内存优化
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 初始化GradScaler用于混合精度训练
    # 这一步解决了以下问题：
    # 1. 数值稳定性：在float16/bfloat16精度下，梯度值可能过小导致下溢，GradScaler通过缩放损失值来避免这个问题
    # 2. 训练效率：使用低精度计算可以加速训练过程，同时减少显存占用
    # 3. 自动管理：GradScaler自动处理损失缩放和梯度反缩放，简化了混合精度训练的实现
    # 4. 条件启用：根据用户指定的dtype参数决定是否启用混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # 使用AdamW优化器初始化模型参数优化
    # 这一步解决了以下问题：
    # 1. 参数优化：AdamW结合了Adam优化器的自适应学习率和权重衰减，能够更有效地优化模型参数
    # 2. 学习率控制：通过设置初始学习率，为模型训练提供合适的参数更新步长
    # 3. 稳定性：AdamW的权重衰减实现更稳定，有助于防止过拟合
    # 4. 收敛性：自适应学习率机制有助于模型更快地收敛到较好的解
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        # 设置需要忽略的分布式参数和缓冲区
        # 这一步解决了以下问题：
        # 1. 特殊参数处理：pos_cis是预计算的位置编码，在分布式训练中不需要同步
        # 2. 性能优化：避免不必要的参数同步，提高分布式训练效率
        # 3. 内存节省：减少分布式通信时的内存开销
        # 4. 正确性保证：确保位置编码在分布式训练中保持一致
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        
        # 将模型包装为分布式数据并行模型
        # 这一步解决了以下问题：
        # 1. 分布式训练：支持多GPU并行训练，加速模型训练过程
        # 2. 数据并行：自动将数据分割到不同GPU，实现并行计算
        # 3. 梯度同步：自动处理不同GPU间的梯度同步，确保参数更新一致
        # 4. 设备管理：指定模型运行的GPU设备，确保资源合理分配
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    # 这一步解决了以下问题：
    # 1. 训练进度跟踪：通过知道每个epoch的总迭代次数，可以准确计算和显示训练进度
    # 2. 学习率调度：在余弦退火等学习率调度策略中，需要知道总步数来正确计算当前学习率
    # 3. 日志记录：在记录训练日志时，可以显示当前step在总迭代次数中的位置
    # 4. 时间预估：结合已用时间，可以估算完成一个epoch或整个训练所需的时间
    # 5. 检查点保存：根据总迭代次数，可以确定在哪些step保存模型检查点
    iter_per_epoch = len(train_loader)
    
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
