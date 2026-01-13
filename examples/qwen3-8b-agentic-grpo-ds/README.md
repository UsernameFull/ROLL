# Qwen3-8B GRPO Agentic Training

使用 ROLL 框架训练 Qwen3-8B 模型，采用 **DeepSpeed** 后端、**GRPO** 算法和 **Agentic** 流水线。

## 配置概览

| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen3-8B |
| 训练后端 | DeepSpeed (ZeRO-2) |
| 推理后端 | vLLM |
| 算法 | GRPO (Group Relative Policy Optimization) |
| 流水线 | Agentic Pipeline |
| GPU 数量 | 8 |

## 快速开始

### Linux/macOS

```bash
# 从 ROLL 项目根目录运行
cd /path/to/ROLL
bash examples/qwen3-8b-agentic-grpo-ds/run_grpo_agentic.sh
```

### Windows (PowerShell)

```powershell
# 从 ROLL 项目根目录运行
cd D:\project\ROLL
.\examples\qwen3-8b-agentic-grpo-ds\run_grpo_agentic.ps1
```

## 关键配置说明

### GRPO 算法参数

- `adv_estimator: "grpo"` - 使用 GRPO 优势估计器
- `num_return_sequences_in_group: 8` - 每个 prompt 生成 8 个响应用于组内比较
- `use_kl_loss: true` - 启用 KL 散度损失
- `kl_loss_coef: 0.001` - KL 损失系数
- `whiten_advantages: true` - 对优势值进行白化处理

### DeepSpeed 配置

使用 ZeRO-2 优化策略（适合 8B 模型）:

```yaml
strategy_args:
  strategy_name: deepspeed_train
  strategy_config: ${deepspeed_zero2}
```

如需使用 ZeRO-3（更大模型或更少显存）:

```yaml
strategy_args:
  strategy_name: deepspeed_train
  strategy_config: ${deepspeed_zero3}
```

### 环境配置

默认包含两个 Agentic 环境:
- **FrozenLake**: 冰面寻路游戏
- **SimpleSokoban**: 推箱子游戏

可以通过修改 `train_env_manager.tags` 和 `custom_envs` 添加更多环境。

## 自定义配置

### 修改学习率

```yaml
actor_train:
  training_args:
    learning_rate: 5.0e-7  # 调整学习率
```

### 调整批次大小

```yaml
rollout_batch_size: 512  # 增加 rollout 批次
actor_train:
  training_args:
    per_device_train_batch_size: 2  # 每个设备的批次大小
    gradient_accumulation_steps: 8   # 梯度累积步数
```

### 使用 Wandb 进行追踪

```yaml
track_with: wandb
tracker_kwargs:
  api_key: <your_api_key>
  project: qwen3-agentic-grpo
  name: ${exp_name}
```

## 文件结构

```
qwen3-8b-agentic-grpo-ds/
├── grpo_agentic_config.yaml  # 主配置文件
├── run_grpo_agentic.sh       # Linux/macOS 启动脚本
├── run_grpo_agentic.ps1      # Windows PowerShell 脚本
└── README.md                  # 说明文档
```

## 依赖检查

确保已安装以下依赖:

```bash
# DeepSpeed
pip install deepspeed

# vLLM (推理)
pip install vllm

# Flash Attention 2
pip install flash-attn --no-build-isolation
```

## 常见问题

### Q: 显存不足怎么办？

A: 尝试以下方案：
1. 减小 `per_device_train_batch_size`
2. 增大 `gradient_accumulation_steps`
3. 使用 `deepspeed_zero3` 配置
4. 降低 `gpu_memory_utilization`

### Q: 如何使用 LoRA 训练？

A: 在 `actor_train.model_args` 中添加 LoRA 配置，并参考 `examples/qwen2.5-7B-rlvr_megatron/rlvr_lora_zero3.yaml`。
