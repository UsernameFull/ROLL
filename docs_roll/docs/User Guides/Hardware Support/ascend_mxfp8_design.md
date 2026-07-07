# Ascend ModelSlim MXFP8 Design

Last updated: 07/07/2026.

本文档说明 ROLL 支持 Ascend ModelSlim MXFP8 已量化模型的设计思路。它关注代码边界和关键决策，不替代 Ascend 环境安装、训练启动或 vLLM 配置教程。

## 1. Goal

ROLL 需要同时支持两类 MXFP8 场景：

- 训练侧：Megatron/MCore 在 Ascend NPU 上加载 ModelSlim MXFP8 预量化 checkpoint，并进行真 FP8 参数训练。
- 推理侧：vLLM 支持从 BF16/FP16 权重 online quant 到 Ascend MXFP8，也支持直接加载预量化 Ascend checkpoint。

核心要求是避免隐式退化。只要用户声明 `fp8_param=True` 和 `quantized_checkpoint_format=ascend_mxfp8`，训练侧就必须加载为可训练 FP8 参数状态；如果 MindSpeed/MindSpeed-TE 无法提供对应能力，ROLL 应该直接报错，而不是反量化成 BF16 后继续跑。

## 2. Key Decisions

### Keep explicit checkpoint format

`fp8_recipe=mxfp8` 只说明训练 recipe，不说明 checkpoint 已经是 ModelSlim 预量化格式。因此仍保留显式声明：

```yaml
quantized_checkpoint_format: ascend_mxfp8
```

这个字段是训练加载分支的开关，也能避免把普通 HF checkpoint 误判成预量化 checkpoint。

### Keep strict `fp8_param` validation

`fp8_param=True` 只允许以下组合：

```yaml
transformer_impl: transformer_engine
fp8: e4m3
fp8_recipe: mxfp8
fp8_param: true
quantized_checkpoint_format: ascend_mxfp8
```

同时平台必须是 Ascend NPU。`TrainingArguments` 和 `McaModelConfig` 都做校验：前者保护用户输入，后者保护从 checkpoint 或 config 文件恢复出来的模型配置。

### Keep loader discovery minimal

ROLL 按 verl 的 Ascend 设计思路优先对齐 MindSpeed/MindSpeed-TE。默认只尝试少数 MindSpeed 风格入口：

```text
mindspeed.core.transformer.custom_layers.transformer_engine.load_modelslim_mxfp8_state_dict
mindspeed.core.transformer.custom_layers.transformer_engine.load_ascend_mxfp8_state_dict
```

如果实际 MindSpeed 版本暴露的 loader 入口不同，推荐提供一个薄 wrapper，并通过环境变量指定：

```bash
ROLL_ASCEND_MXFP8_STATE_DICT_LOADER=package.module.load_fn
```

wrapper 对 ROLL 暴露稳定签名：

```python
def load_fn(*, model_path, config, converter, vp_stage, adapter):
    ...
    return state_dict
```

这样第三方包内部 API 变化时，只需要调整 wrapper，不需要让 ROLL 核心代码维护一组猜测路径。

## 3. Training Path

训练侧链路如下：

```text
TrainingArguments
    -> strict fp8_param validation
McaModelConfig
    -> checkpoint format validation
PretrainedModel.from_pretrained()
    -> detect quantized_checkpoint_format=ascend_mxfp8
AscendMxfp8CheckpointAdapter
    -> parse ModelSlim quant description
    -> pair qweight and scale sidecar
    -> keep FLOAT weights as floating tensors
MindSpeed-TE loader
    -> materialize trainable FP8 parameter state
Megatron model.load_state_dict()
```

`AscendMxfp8CheckpointAdapter` 只负责格式识别和轻量校验：

- 读取 `quant_model_description.json`。
- 或读取 HF `config.json.quantization_config`。
- 识别 `quant_method=ascend`。
- 区分 `W8A8_MXFP8` 权重和 `FLOAT` 权重。
- 为量化权重寻找 scale sidecar。

它不负责 QKV merge/split、gate_up merge/split、TP/EP shard 或 TE 参数状态构造。原因是这些转换会改变权重和 scale 的对应关系，必须由真正理解 MindSpeed-TE FP8 参数状态的 loader 来处理。

找不到 loader 时，ROLL 明确报错：

```text
ROLL will not silently dequantize to BF16.
```

这是训练侧最重要的安全边界。

## 4. Inference Path

推理侧分两条路径。

### Online MXFP8

用于训练权重仍是 BF16/FP16，但 vLLM rollout 希望用 Ascend MXFP8 的场景：

```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      online_quantization: ascend_mxfp8
      online_quantization_config:
        group_size: 32
```

ROLL 会生成 ModelSlim 风格的 `hf_overrides.quantization_config`，设置：

```yaml
quantization: ascend
load_format: dummy
```

随后 vLLM worker 在权重同步时调用 `torch_npu.npu_dynamic_mx_quant()`，把浮点权重量化为 MXFP8 qweight + scale。

### Pre-quantized Ascend checkpoint

用于 checkpoint 已经由 ModelSlim 量化好的场景：

```yaml
actor_infer:
  strategy_args:
    strategy_name: vllm
    strategy_config:
      quantization: ascend
      load_format: auto
```

如果用户未显式设置 `load_format`，ROLL 对 `quantization=ascend` 默认使用 `auto`，让 vLLM-Ascend 直接加载预量化 checkpoint。

训练到推理的权重同步仍默认走 BF16/master/dequantized 视图。也就是说，训练侧可以是 TE FP8 参数，推理侧更新时仍先拿到普通浮点 HF 权重，再由 vLLM worker 侧量化。这降低了训练框架和推理框架之间的耦合。

## 5. Boundaries and TODO

当前设计刻意不覆盖：

- FSDP2/HF 真 FP8 参数训练。
- SGLang 预量化 Ascend MXFP8 checkpoint 的一等加载入口。
- ROLL 内部完整实现 ModelSlim MXFP8 到 TE 可训练参数状态的转换。
- 将训练 checkpoint 导出为 ModelSlim MXFP8。

后续优先补齐：

1. `AscendMxfp8CheckpointAdapter` 增加未知 quant type 校验。
2. 增加 qweight/scale shape mismatch 校验。
3. 在真实 Ascend 环境确认 MindSpeed/MindSpeed-TE loader API。
4. 提供推荐 wrapper 示例，稳定 `ROLL_ASCEND_MXFP8_STATE_DICT_LOADER` 接口。
5. 增加 NPU 集成测试：load small ModelSlim checkpoint、forward/backward、optimizer step、save/resume、同步到 vLLM generation。

## Source Map

关键文件：

- `mcore_adapter/src/mcore_adapter/training_args.py`
- `mcore_adapter/src/mcore_adapter/models/model_config.py`
- `mcore_adapter/src/mcore_adapter/models/model_factory.py`
- `mcore_adapter/src/mcore_adapter/models/converter/ascend_mxfp8.py`
- `mcore_adapter/src/mcore_adapter/npu_runtime.py`
- `roll/utils/fp8.py`
- `roll/utils/vllm_online_quantization.py`
- `roll/third_party/vllm/fp8.py`
- `roll/third_party/vllm/worker.py`

外部参考：

- verl Ascend Dockerfile: https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3
- verl MindSpeed engine: https://github.com/verl-project/verl/blob/main/verl/workers/engine/mindspeed/transformer_impl.py
- verl vLLM FP8 utilities: https://github.com/verl-project/verl/blob/main/verl/utils/vllm/vllm_fp8_utils.py
