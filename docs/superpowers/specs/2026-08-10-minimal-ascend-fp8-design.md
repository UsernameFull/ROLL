# 昇腾 FP8 最小化支持设计

## 目标

在缩小当前分支相对 `alibaba/main` 差异的同时，保留以下三条端到端昇腾
FP8 能力：

1. 使用 BF16/FP16 模型参数进行 Megatron FP8 计算训练。
2. vLLM 将同步得到的 BF16/FP16 权重在线转换为昇腾 MXFP8。
3. Megatron 从 ModelSlim 预量化 MXFP8 checkpoint 加载并训练。

实现只面向以下固定依赖组合：

- vLLM `0.23.0`
- vLLM-Ascend `0.23.0rc1`
- MegatronAdaptor `core_r0.17.0`
- 与 MegatronAdaptor 配套的 TransformerEngineNPU

支持这些固定版本优先于兼容更早的 vLLM 或 vLLM-Ascend 版本。

## 当前状态

审计开始时，当前分支相对 `alibaba/main` 领先 48 个提交，包含 6,305 行新增
和 406 行删除。差异中混合了 FP8 功能、NPU 基础适配、旧版 vLLM 兼容、
SGLang 和 FSDP2 实验、诊断代码、CI、容器配置、示例以及历史设计文档。
其中最大的可裁剪部分是 vLLM FP8 和 worker 补丁中的多版本分支与参数兼容代码。

## 架构设计

### Megatron FP8 计算训练

`TrainingArguments` 作为公开配置入口，接收 Megatron FP8 格式和 recipe，校验
受支持的参数组合，并仅通过 `megatron_strategy` 将必要参数传递给
`McaModelConfig`。

在 NPU 环境中，初始化流程必须在构建模型前导入 MegatronAdaptor，应用其默认值，
并将 ROLL 参数同步给 adaptor。FP8 autocast、recipe、缩放和可训练状态由原生
Megatron 与 TransformerEngineNPU 负责，ROLL 不重复实现这些算法。

### vLLM 在线 MXFP8

vLLM strategy 接受 `online_quantization: ascend_mxfp8`。配置推理引擎时，ROLL
为支持的 Qwen 稠密模型和 MoE 模型生成最小的 ModelSlim 兼容量化描述，并作为
昇腾量化配置传入 vLLM。

更新权重时，vLLM 首先执行正常的 TP/EP 分片。随后 ROLL worker hook 使用
`torch_npu.npu_dynamic_mx_quant` 量化每个符合条件的浮点权重分片，提供对应的
scale tensor，再将权重打包和运行时准备交给 vLLM-Ascend 0.23 的量化方法。
重复更新必须在不改变图执行所依赖参数身份的前提下替换量化值。

不再保留 vLLM 0.10–0.20 的权重处理实现或版本选择逻辑。

### ModelSlim 预量化训练

通过显式配置 `quantized_checkpoint_format: ascend_mxfp8` 选择该路径。轻量
checkpoint adapter 负责读取 ModelSlim 量化元数据、区分量化权重与浮点权重，
并为每个量化权重匹配对应的 scale sidecar。

adapter 不自行实现 QKV 合并、MoE 权重合并或分布式分片，而是通过一个固定的
MindSpeed-TE loader 接口完成这些操作。如果 loader 不可用，或者 weight/scale
不完整，加载过程必须抛出明确错误，禁止静默反量化为 BF16 后继续训练。

## 最小化规则

只有位于上述三条运行链路中，或被这些链路的聚焦测试直接依赖的文件，才允许保留
相对 `alibaba/main` 的改动。

删除以下内容：

- vLLM 0.23 之前版本的分支与实现；
- SGLang、FSDP2、CUDA/H100 和无关模型扩展；
- 诊断日志以及用于排查 log probability 的代码；
- 固定上游版本已经提供的参数子类和运行时兼容抽象；
- 重复示例、历史设计文档，以及运行时不依赖的 FP8 专用 CI 脚手架。

保留以下内容：

- 一个同时覆盖 Megatron FP8 和 vLLM 在线 MXFP8 的四 NPU 示例；
- 配置、元数据解析、量化和重复权重更新的聚焦回归测试；
- 此前要求修复的 MoE 分组计数、NPU RNG、短响应 whitening 和 GDN worker
  问题；
- 用户现有的 Dockerfile 改动和未跟踪的 BF16 示例。

通用 NPU 改动只有在四 NPU FP8 链路直接依赖时才保留。仅仅对其他 NPU 模型或
框架有帮助，不构成保留理由。

## 错误处理

- 启动时拒绝不受支持的依赖版本，并在错误信息中列出预期版本组合。
- 拒绝相互冲突的 `quantization` 和 `online_quantization` 配置。
- 除非 NPU、TransformerEngineNPU、MXFP8 recipe 和显式 ModelSlim checkpoint
  格式同时启用，否则拒绝 `fp8_param=True`。
- 拒绝缺失或格式错误的量化元数据，以及缺失 scale tensor 的 checkpoint。
- 缺少 MindSpeed-TE 可训练 loader 时直接报错，不回退为反量化参数。

## 验证方案

可在 CPU 环境运行的测试覆盖配置校验、量化描述生成、checkpoint 元数据解析、
weight/scale 配对，以及使用 mock 量化算子的重复更新生命周期。

NPU smoke test 使用保留的 Qwen3 0.6B 四 NPU 示例，并验证：

1. Megatron 创建启用了 FP8 的 TransformerEngineNPU 层。
2. 一次 forward、backward 和 optimizer step 正常完成。
3. BF16/master 权重成功同步到 vLLM worker。
4. worker 在分片后生成 MXFP8 权重与 scale。
5. 连续执行两次权重更新和一次 generation 均能完成。

最终审查将统计相对 `alibaba/main` 的改动文件和行数，确认每项剩余差异都能映射
到已批准的运行链路，并明确报告本地环境无法执行的验证项。
