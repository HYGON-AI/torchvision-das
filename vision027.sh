#!/usr/bin/env bash

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ALIYUN_INDEX="http://mirrors.aliyun.com/pypi/simple/"

export ROCM_PATH="${ROCM_PATH:-/opt/dtk}"
# shellcheck disable=SC1091
source "${ROCM_PATH}/env.sh"
if [[ -f "${ROCM_PATH}/cuda/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ROCM_PATH}/cuda/env.sh"
fi

if [[ -n "${EXPECTED_PYTHON_VERSION:-}" ]]; then
    python3 -c 'import os, sys; expected = os.environ["EXPECTED_PYTHON_VERSION"]; actual = f"{sys.version_info.major}.{sys.version_info.minor}"; assert actual == expected, f"expected Python {expected}, got {actual}"'
fi

dnf install -y libjpeg-turbo-devel libwebp-devel
python3 -m pip install --no-cache-dir \
    numpy==1.26.4 wheel setuptools ninja \
    -i "${ALIYUN_INDEX}" --trusted-host mirrors.aliyun.com
python3 -m pip install --no-cache-dir --pre --upgrade --force-reinstall \
    "torch==2.12.0" flash_attn

cd "${REPO_ROOT}"
export MAX_JOBS="${MAX_JOBS:-32}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-${AMDGPU_TARGETS:-}}"
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --force-reinstall dist/torchvision-*.whl

python3 -m pip install --no-cache-dir \
    pytest pytest-mock pytest-xdist torchdata pytest-time pytest-timeout pytest-rerunfailures \
    -i "${ALIYUN_INDEX}" --trusted-host mirrors.aliyun.com

export PYTORCH_TEST_WITH_ROCM=1
export CONTINUE_THROUGH_ERROR=True
export MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC=1
export ROCBLAS_ATOMICS_MOD=0
export MIOPEN_DEBUG_HIPNN_CONV_S591=0
export PYTORCH_TEST_WITH_EXTENDED=0
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-1,2}"
export FA_SO_PATH="$(python3 - <<'PY'
import glob
import site

matches = []
for directory in site.getsitepackages():
    matches.extend(glob.glob(f"{directory}/flash_attn_2_cuda*.so"))
if not matches:
    raise SystemExit("flash_attn_2_cuda shared library not found")
print(matches[0])
PY
)"

PYTHONHASHSEED=0 python3 -m pytest -vv "${REPO_ROOT}/test/" \
    --ignore="${REPO_ROOT}/test/test_prototype_datasets_builtin.py" \
    --ignore="${REPO_ROOT}/test/test_prototype_datasets_utils.py" \
    --ignore="${REPO_ROOT}/test/test_prototype_models.py" \
    --ignore="${REPO_ROOT}/test/test_prototype_transforms.py" \
    --ignore="${REPO_ROOT}/test/test_datasets_download.py" \
    -n 4 --dist=worksteal \
    --timeout=1800 --timeout-method=thread \
    -ra
