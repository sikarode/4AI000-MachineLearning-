#!/usr/bin/env bash

# setup_env.sh — 创建并配置 Python 环境
#

set -e

ENV_NAME=${1:-ml4phys}
PYTHON_VERSION=3.10

# ------------------------------------------------------------------
# 1. 创建/激活环境
# ------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    echo "使用 conda 创建环境 '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION
    echo "激活 conda 环境" $ENV_NAME
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "未检测到 conda，改用 venv 创建环境" >&2
    python$PYTHON_VERSION -m venv "$ENV_NAME"
    # shellcheck disable=SC1090
    source "$ENV_NAME/bin/activate"
fi

# ------------------------------------------------------------------
# 2. 升级 pip 并安装依赖
# ------------------------------------------------------------------
python -m pip install --upgrade pip

# CUDA-aware torch install: 用户可根据 GPU 版本调整
echo "安装 PyTorch（CPU/GPU 自动检测）..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 || \
python -m pip install torch torchvision torchaudio

# 安装项目其他依赖
python -m pip install diffusers tqdm numpy

# 可选但推荐的工具
python -m pip install transformers accelerate scikit-learn matplotlib

# 将依赖锁定到 requirements.txt
python - <<'PY'
import pkg_resources, json, sys
reqs = sorted(str(r) for r in pkg_resources.working_set)
open('requirements.txt','w').write("\n".join(reqs))
print('已生成 requirements.txt')
PY

# ------------------------------------------------------------------
# 3. 提示用户下一步
# ------------------------------------------------------------------
echo
echo "环境 '$ENV_NAME' 已准备就绪。"
echo "在此环境中，您可以运行："
echo "    python train.py --data_dir ./data --epochs1 200 --epochs2 200 --batch_size 8"
echo "或者进行推理： python inference.py ..." 

echo "若您使用 conda，请通过 'conda deactivate' 退出环境。"

echo "注意：Windows 用户可在 PowerShell 运行 'bash setup_env.sh' 或直接手动执行这些命令。"
