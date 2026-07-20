#!/usr/bin/env bash

IMAGE_NAME="${1:-roll-cann901-a5:latest}"
CONTAINER_NAME="${2:-roll-cann901-a5}"
DATA_DIR="${3:-/data}"

if [ ! -e /dev/davinci0 ] || [ ! -e /dev/davinci1 ] || [ ! -e /dev/davinci2 ] || [ ! -e /dev/davinci3 ]; then
    echo "未检测到完整的 0-3 号 NPU 设备。"
    exit 1
fi

if [ ! -e /dev/davinci_manager ] || [ ! -e /dev/hisi_hdc ]; then
    echo "未检测到 Ascend 管理设备。"
    exit 1
fi

if [ ! -d /usr/local/Ascend/driver/topo ]; then
    echo "未找到 /usr/local/Ascend/driver/topo，无法保证 HCCL 通信正常。"
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    echo "数据目录不存在：${DATA_DIR}"
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "容器已存在：${CONTAINER_NAME}"
    exit 1
fi

docker run -dit \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --security-opt label=disable \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /usr/local/Ascend/driver/topo:/usr/local/Ascend/driver/topo:ro \
    -v "${DATA_DIR}":/data \
    "${IMAGE_NAME}" \
    bash

docker exec "${CONTAINER_NAME}" bash -lc 'source /workspace/roll_env.sh && python3 -c "import torch, torch_npu; print(f\"torch={torch.__version__} torch_npu={torch_npu.__version__} npu_count={torch.npu.device_count()}\")"'

echo "容器已创建：${CONTAINER_NAME}"
echo "进入容器：docker exec -it ${CONTAINER_NAME} bash"
