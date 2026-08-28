#!/bin/bash

set -euo pipefail

# -------------------- Network Configuration --------------------
NET_TYPE="high"
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
export NCCL_MPI_PROFILE_PRIMS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi

export DISABLE_VERSION_CHECK=1

# -------------------- Node Configuration --------------------
export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "10.0.0.1,10.0.0.2" or single node "127.0.0.1"
export IP_LIST=${IP_LIST:-"127.0.0.1"}

MASTER_PORT=${MASTER_PORT:-29500}

IFS=',' read -ra IP_ARRAY <<< "$IP_LIST"
NODES=${#IP_ARRAY[@]}
MASTER_ADDR=${IP_ARRAY[0]}

# -------------------- Paths --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="${YAML_FILE:-hy_v4_full_sft.yaml}"
# If YAML_FILE is not an absolute path, resolve it relative to SCRIPT_DIR
if [[ "${YAML_FILE}" != /* ]]; then
    YAML_FILE="${SCRIPT_DIR}/${YAML_FILE}"
fi
ENTRY_SCRIPT="${SCRIPT_DIR}/train_hy_v4.py"

# -------------------- Distributed Environment --------------------
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NNODES="${NODES}"

# Prevent Accelerate/FSDP from upcasting bf16 parameters back to fp32
export ACCELERATE_MIXED_PRECISION=no

if [ ${NODES} -gt 1 ]; then
    # Determine local node rank by matching local IP against IP_LIST
    LOCAL_IP=$(hostname -i | awk '{print $1}')
    NODE_RANK=0
    for i in "${!IP_ARRAY[@]}"; do
        if [[ "${IP_ARRAY[$i]}" == "${LOCAL_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done
    export RANK="${NODE_RANK}"
else
    export RANK=0
fi

echo "============================================"
echo "  HYV4 LLaMA Factory Training"
echo "  Nodes: ${NNODES}, Rank: ${RANK}"
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs per node: ${HOST_GPU_NUM}"
echo "  Total GPUs: $((NODES * HOST_GPU_NUM))"
echo "============================================"

# -------------------- Launch --------------------
# We launch torchrun directly (instead of FORCE_TORCHRUN) so that each
# worker process runs train_hy_v4.py with all HYV4 patches applied.
torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${RANK}" \
    --nproc_per_node "${HOST_GPU_NUM}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${ENTRY_SCRIPT}" "${YAML_FILE}"
