# ROLL CANN 9.0.1 A5 容器设计

## 目标

提供可复现的 ROLL Ascend 950 四卡开发/训练容器。镜像构建阶段从 GitHub 获取并固定 ROLL、Megatron-LM、TransformerEngineNPU、MegatronAdaptor、vLLM 和 vLLM-Ascend 的源码版本；启动阶段挂载宿主机驱动、NPU 设备、驱动拓扑和数据目录。

## 版本组合

| 组件 | 版本或提交 |
| --- | --- |
| 基础镜像 | `cann:9.0.1-950-ubuntu22.04-py3.12` |
| torch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post2` |
| triton-ascend | `3.2.1` |
| vLLM | `v0.23.0`，以 `VLLM_TARGET_DEVICE=empty` 构建 |
| vLLM Ascend | `v0.23.0rc1` |
| ROLL | 构建时获取最新 `npu_fp8` 分支 |
| Megatron-LM | `core_r0.17.0@963bf39218e8bb83a1203b40293358498322be50` |
| TransformerEngineNPU | `48378a9cefdba637eb3f457badb043ec4c1d622e` |
| MegatronAdaptor | `core_r0.17.0@d2fb23d75b8f7875079dfa3b52a5e4429d45aa9d` |

## 镜像构建

Dockerfile 设置华为云 PyPI 源，单独从 PyTorch CPU 索引安装带 `+cpu` 后缀的 torch。torch-npu 以 `--no-deps` 安装，避免普通 PyPI 源无法解析 `torch==2.10.0+cpu`。

vLLM 与 vLLM-Ascend 使用 editable 模式安装。vLLM-Ascend 的 `arctic-inference` 依赖不安装：它会触发不匹配的 torch 下载，且 Qwen/RLVR 路径不使用该可选模型加载器。文本 ROLL 用例使用 `numpy==1.26.4`；该固定版本是 triton-ascend 3.2.1 的要求。

镜像在 `/etc/profile.d/roll.sh` 写入 CANN、NNAL、ROLL、ModelScope、HCCL 和 Ascend 950 的环境变量。登录 shell 自动加载；非登录 shell 可显式执行 `source /workspace/roll_env.sh`。

## 启动与挂载

启动脚本将 `/dev/davinci0` 至 `/dev/davinci3`、`/dev/davinci_manager` 和 `/dev/hisi_hdc` 传入容器，并挂载 CANN 驱动运行库、DCMI、npu-smi、驱动版本和安装信息。

必须挂载 `/usr/local/Ascend/driver/topo` 到相同容器路径。CANN 9.0.1 HCCL 会读取 `topo/950/atlas_350_3.json`；缺失时四卡 all-reduce 会报 `RootInfoDetect` 和错误码 4。

NNAL 已包含在所选 CANN 镜像中，因此不额外挂载 NNAL。脚本以 `--network host`、`--ipc host` 运行并挂载宿主机 `/data` 到容器 `/data`。

## 验证

容器启动后运行：

```bash
source /workspace/roll_env.sh
python3 -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())'
torchrun --standalone --nproc_per_node=4 /workspace/hccl_smoke.py
```

预期设备数量为 4，HCCL 规约的四个 rank 均输出 `all_reduce=10.0 OK`。
