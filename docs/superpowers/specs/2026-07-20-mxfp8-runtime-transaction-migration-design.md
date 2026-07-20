# MXFP8 运行时事务迁移设计

## 目标

将旧容器中已验证的 Ascend MXFP8 运行时修复迁移到本地 ROLL 工作区，同时排除临时 BF16 用例、备份文件和实验目录。

## 范围

保留以下最小改动：

1. 在 vLLM 推理模型更新前后建立事务边界：开始时恢复已转换的 MXFP8 参数，完成所有分桶加载后只转换一次。
2. 为 `AscendW8A8MXFP8DynamicLinearMethod` 注入权重加载器：加载非 FP8 权重时在 NPU 上执行 per-block MXFP8 量化，并写入权重 scale。
3. 在指标聚合前递归将 NPU tensor 移至 CPU，保证 NumPy 聚合可用。
4. 为分桶模型更新事务增加单元测试。

## 排除项

不迁移 BF16 配置和启动脚本、`rl_examples`、`.bak-*` 文件，以及 `Worker` 基类上的同步 `begin_model_update` 入口。事务 RPC 链路仅覆盖 `InferWorker`、`VllmStrategy`、`CustomAsyncLLM`、`CustomAsyncLLMEngine` 和 `WorkerBase`。

## 验证

运行 `tests/third_party/vllm/test_mxfp8_runtime_refs.py`。测试应验证：事务开始只恢复一次、多个权重桶加载期间不执行 finalize、重复开始事务会失败、事务完成时只 finalize 一次。
