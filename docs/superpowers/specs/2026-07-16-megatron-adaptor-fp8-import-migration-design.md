# MegatronAdaptor FP8 导入迁移设计

## 目标

将昇腾 Megatron FP8 运行时从旧版 MindSpeed 包结构迁移到 `npu_ci_all` 已采用的独立
`MegatronAdaptor core_r0.17.0` 与 `TransformerEngineNPU` 技术栈。

迁移必须保留现有 FP8 参数推导和 Transformer Engine 集成行为，同时移除所有来自 `mindspeed.*` 的运行时
导入。在 NPU Megatron 初始化期间，如果缺少独立依赖，必须给出清晰错误并立即失败，不能静默禁用 NPU
补丁。

## 范围

本次修改包括：

- NPU Megatron 运行时引导和适配器模块状态跟踪。
- MegatronAdaptor 功能默认值与 FP8 参数同步。
- 通过 Megatron 标准接口发现 TransformerEngineNPU 符号。
- Megatron NPU 测试以及 NPU CI 依赖栈。
- 将仍然使用 MindSpeed 含义的内部名称和消息调整为 MegatronAdaptor 含义。

现有 MXFP8 模型更新事务和 vLLM 权重加载修改保持不变。

## 非目标

- 在同一个版本中同时支持 `mindspeed.megatron_adaptor` 和独立 `megatron_adaptor`。
- 独立依赖不可用时回退到旧版 MindSpeed 包。
- 修改 FP8 格式、recipe、checkpoint 转换或模型更新语义。
- 重构无关的 NPU 算子补丁或通用 Megatron 初始化逻辑。

## 运行时设计

### 严格加载独立适配器

在 NPU 上，Megatron 初始化会在模型并行初始化和 FP8 模型构建之前导入 `megatron_adaptor`。已导入的模块是
判断适配器是否加载以及执行重新打补丁的唯一依据。

该引导过程在非 NPU 平台上保持为空操作。在 NPU 上，如果导入失败，则抛出明确指出需要
`MegatronAdaptor core_r0.17.0` 的 `RuntimeError`。运行时不尝试导入 `mindspeed.megatron_adaptor`。

```text
initialize_megatron(args)
  -> bootstrap_npu_runtime()
       -> import torch_npu
       -> import megatron_adaptor
       -> 校验标准 Transformer Engine 集成
       -> 安装现有 NPU RNG 兼容钩子
  -> sync_megatron_adaptor_args(args)
  -> 初始化分布式和模型并行通信组
```

### FP8 参数同步

保留现有派生参数行为。运行时参数从
`megatron_adaptor.utils.args_utils.get_mindspeed_args` 获取；上游辅助函数虽然已移动到新包，但函数名保持不变。

ROLL 继续同步共有字段，并派生以下 FP8 和注意力参数：

- `fp8` 和 `fp8_format` 继续作为别名。
- 启用 FP8 且未显式配置时，设置 `transformer_impl="transformer_engine"`。
- `use_flash_attn_npu_batch_invariant=True` 时禁用 `use_flash_attn`。
- 在 NPU 上使用 Transformer Engine 且未选择 batch-invariant attention 时，默认启用 `use_flash_attn`。

如果初次打补丁后同步值发生变化，运行时调用 `megatron_adaptor.repatch(updates)`。重新打补丁属于兼容性刷新，
失败时沿用现有行为记录警告；适配器导入失败或必需符号缺失仍然是致命错误。

ROLL 内部 API 从 `mindspeed` 命名迁移到 `megatron_adaptor` 命名。上游 `get_mindspeed_args` 函数名只在
运行时边界内使用，不对外包装或重命名。

### TransformerEngineNPU 集成

加载 `megatron_adaptor` 后，ROLL 只通过 `megatron.core.extensions.transformer_engine` 获取 Transformer
Engine 类和 checkpoint 辅助函数，不再从 `mindspeed.core.*` 导入 `TENorm` 或已打补丁的模型模块。

ROLL 保留现有 `_npu_te_checkpoint` 调用签名适配器，用于 Qwen3-VL 激活 checkpoint。MegatronAdaptor 引导
完成后，仅当标准 `megatron.core.extensions.transformer_engine` 模块缺少 `te_checkpoint` 时，才将该适配器
发布到这个标准模块。运行时不再创建或修改 `transformer_engine.*`、`mindspeed.core.*` 模块。独立适配器、
TransformerEngineNPU 未提供所需 Transformer Engine 模块或 `TENorm` 时，FP8 初始化抛出明确依赖错误。

该边界使模型代码不依赖供应商包内部目录结构，并与 MegatronAdaptor 0.17 的集成方式保持一致。

## 代码修改

### `mcore_adapter/src/mcore_adapter/npu_runtime.py`

- 将旧适配器导入器和模块加载检查替换为独立 `megatron_adaptor` 实现。
- 将 `mindspeed.args_utils` 替换为 `megatron_adaptor.utils.args_utils`。
- 通过已加载的 `megatron_adaptor` 模块执行 `repatch`。
- 移除所有 `mindspeed.core.*` 模块补丁目标和导入。
- 重命名仍将集成描述为 MindSpeed 的内部常量和函数。
- 缺少必需的独立包或 FP8 Transformer Engine 符号时，抛出清晰运行时错误。

### `mcore_adapter/src/mcore_adapter/initialize.py`

- 导出并调用重命名后的 MegatronAdaptor 功能默认值和参数同步辅助函数。
- 保持“先引导运行时，再初始化分布式环境”的顺序。

### `mcore_adapter/src/mcore_adapter/models/model_config.py`

- 通过重命名后的 MegatronAdaptor 辅助函数应用功能默认值。
- 保留现有 FP8、Transformer Engine 校验和注意力参数推导。

### 测试与 CI

- 将 Megatron NPU 测试改为导入 `megatron_adaptor`。
- 增加针对以下行为的运行时测试：非 NPU 空操作、严格的 NPU 导入失败、独立包参数发现、FP8 派生参数
  同步以及重新打补丁。
- 参照 `npu_ci_all`，原地更新 `.github/workflows/ci-npu-mindspeed.yml`，安装并校验 Megatron-Core 0.17、
  MegatronAdaptor `core_r0.17.0` 和 TransformerEngineNPU。
- 保留现有工作流文件名，避免无关的仓库接线改动；显示名称、输入项、缓存键和测试步骤名称改用
  MegatronAdaptor 术语。

## 错误处理

- 非 NPU 执行不要求也不导入 MegatronAdaptor。
- NPU 上缺少 `megatron_adaptor` 时，在分布式初始化之前抛出 `RuntimeError`。
- 缺少 `megatron_adaptor.utils.args_utils` 视为安装了不兼容的 MegatronAdaptor，并抛出说明依赖问题的
  `RuntimeError`。
- FP8 模式缺少必需的 TransformerEngineNPU 符号时，抛出明确指出 TransformerEngineNPU 是预期提供者的
  `RuntimeError`。
- `repatch` 异常继续记录为警告，并包含尝试应用的更新值，保证问题可诊断且不掩盖最初的初始化结果。

## 兼容性

- 支持的 NPU 技术栈：Megatron-Core 0.17.x、MegatronAdaptor `core_r0.17.0` 和 TransformerEngineNPU。
- 明确不再支持旧版 `mindspeed.megatron_adaptor` 环境。
- CUDA 和 CPU 初始化行为不变。
- 当前 Megatron FP8 配置键和 YAML 文件继续有效。
- 当前 vLLM MXFP8 模型更新事务修改保持不变。

## 验证

自动验证包括：

1. 静态搜索，确认运行时和测试不再导入 `mindspeed.*`。
2. 严格独立包引导、参数同步、重新打补丁和非 NPU 隔离的单元测试。
3. 现有 Megatron FP8 别名和 Transformer Engine 选择配置测试。
4. 对修改后的 Python 模块执行编译检查。
5. 使用独立 0.17 技术栈运行 NPU CI。

硬件验收要求 Megatron FP8 模型能够在昇腾设备上完成初始化和一个训练步骤，并在未安装旧版 MindSpeed
包的环境中完成一次到现有 MXFP8 rollout 链路的权重刷新。
