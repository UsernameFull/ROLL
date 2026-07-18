# ROLL FP8 四卡用例设计

## 目标

为现有 Megatron FP8 + vLLM Ascend MXFP8 用例增加基于 Qwen3-0.6B 的可复现四卡版本，并保留原八卡用例。
新用例用于仅暴露 NPU 0 至 NPU 3 的单节点环境。

## 方案

新增独立的四卡 YAML 和启动脚本，不覆盖八卡文件。这样八卡和四卡环境都能直接运行，且无需依赖容易遗漏的
Hydra 命令行覆盖项。

新增文件：

- `examples/ascend_examples/qwen3_0_6b_rlvr_megatron_fp8_mxfp8_4npu.yaml`
- `examples/ascend_examples/run_qwen3_0_6b_rlvr_mxfp8_graph_pipeline_4npu.sh`

四卡 YAML 以现有八卡 YAML 为基线，只修改以下资源字段：

- `num_gpus_per_node` 从 8 调整为 4。
- `actor_train.device_mapping`、`actor_infer.device_mapping` 和 `reference.device_mapping` 从
  `list(range(0,8))` 调整为 `list(range(0,4))`。
- `rewards.math_rule.world_size` 从 8 调整为 4。
- `exp_name` 增加四卡标识，避免与八卡运行目录和 checkpoint 冲突。
- `pretrain` 和 `reward_pretrain` 使用 `Qwen/Qwen3-0.6B`，缩短首次运行的模型下载和初始化时间。

以下配置保持不变：

- Megatron tensor parallel size 为 2，pipeline、expert 和 expert tensor parallel size 为 1。
- Megatron FP8 格式为 `e4m3`，recipe 为 `mxfp8`。
- vLLM tensor parallel size 为 2，在线量化方式为 `ascend_mxfp8`。
- 数据集、训练超参数和 ModelScope 设置。

## 启动流程

四卡启动脚本继续设置现有 HCCL 和 vLLM Ascend 环境变量，并将 Hydra `config_name` 指向新增四卡 YAML。
在远程容器中从仓库根目录执行脚本。

启动前必须确认：

1. 容器能看到 NPU 0 至 NPU 3。
2. 四进程 `torchrun` HCCL smoke test 能完成 barrier 和 all-reduce。
3. ROLL、Megatron-Core、MegatronAdaptor、TransformerEngineNPU、vLLM 和 vLLM Ascend 可导入。
4. 模型和数据集路径可访问。

HCCL smoke test 失败时不启动训练。应先使用 HCCL 日志定位容器拓扑配置问题，采用宿主机已有的有效拓扑文件或
官方生成工具输出；不手工猜测拓扑内容。

## 错误处理

- 配置中发现任何超出 NPU 0 至 NPU 3 的映射时，视为四卡配置错误。
- `num_gpus_per_node`、角色映射和 reward world size 不一致时，不启动远程示例。
- HCCL 通信失败时保留测试日志并停止，避免进入 ROLL 多角色初始化后产生干扰性级联错误。
- 依赖或模型数据缺失时报告具体缺项，不自动替换版本或下载不同模型。

## 验证

本地验证包括：

1. 使用 Python YAML 解析器加载新增配置。
2. 检查四卡资源字段均为 4，所有 device mapping 只覆盖 0 至 3。
3. 检查 TP=2、FP8 `e4m3`、recipe `mxfp8` 和 `ascend_mxfp8` 未发生变化。
4. 检查启动脚本引用新增配置名，并通过 Bash 语法检查。
5. 确认原八卡 YAML 和启动脚本未修改。

远程验收包括：

1. 四卡 HCCL smoke test 成功。
2. ROLL FP8 示例完成各角色初始化，不出现 HCCL `create_config` 或 topology 错误。
3. 至少进入首个 rollout/训练步骤；若受模型或数据外部条件阻塞，保留完整日志并明确阻塞点。
