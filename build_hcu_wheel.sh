#!/usr/bin/env bash

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly HUAWEI_INDEX="https://mirrors.huaweicloud.com/artifactory/pypi-public/simple"
command -v python3 >/dev/null || { echo 'Python 3 is required' >&2; exit 1; }
[[ -f /opt/dtk/env.sh ]] || { echo 'DTK is required at /opt/dtk' >&2; exit 1; }
export ROCM_PATH=/opt/dtk
set +u
# shellcheck disable=SC1091
source /opt/dtk/env.sh
set -u
command -v aicc >/dev/null || { echo 'aicc is required' >&2; exit 1; }
python3 -c 'import torch; assert torch.version.hip, "HCU/ROCm torch is required"; print(f"torch={torch.__version__}")'

cd "${REPO_ROOT}"
git submodule update --init --recursive

dnf install -y libjpeg-turbo-devel libwebp-devel
python3 -m pip install --no-cache-dir \
    numpy==2.1.0 pillow pybind11 wheel setuptools ninja auditwheel patchelf \
    --default-timeout 1000 \
    -i "${HUAWEI_INDEX}"

export USE_MKLDNN=0
export MAX_JOBS="${MAX_JOBS:-32}"
export HIPCC_VERBOSE=0
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-${AMDGPU_TARGETS:-}}"
export LD_LIBRARY_PATH="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"

rm -rf "${REPO_ROOT}/build" "${REPO_ROOT}/dist"
mkdir -p "${REPO_ROOT}/dist/manylinux"

if USE_GPU_VIDEO_DECODER=1 \
    USE_CPU_VIDEO_DECODER=1 \
    python3 setup.py bdist_wheel >/tmp/torchvision-build.log 2>&1; then
    echo "torchvision wheel build succeeded"
    ls -lh dist/torchvision-*.whl
else
    rc=$?
    echo "torchvision wheel build failed; showing last 300 lines"
    tail -n 300 /tmp/torchvision-build.log
    exit "${rc}"
fi

PLATFORM="$(ldd --version | awk 'NR == 1 {split($NF, version, "."); print "manylinux_" version[1] "_" version[2] "_x86_64"}')"
echo "manylinux platform: ${PLATFORM}"
auditwheel repair \
    --plat "${PLATFORM}" \
    --strip \
    --exclude libgalaxyhip.so.5 \
    --exclude libMIOpen.so.1 \
    --exclude librccl.so.1 \
    --exclude libhipblas.so.0 \
    --exclude libhipblas.so.2 \
    --exclude libhipblaslt.so.0 \
    --exclude libhipfft.so \
    --exclude libhipfft.so.0 \
    --exclude libhiprand.so.1 \
    --exclude libhipsolver.so.0 \
    --exclude libhipsparse.so.0 \
    --exclude libhipsparse.so.1 \
    --exclude libhipnn.so \
    --exclude librocblas.so.0 \
    --exclude librocblas.so.4 \
    --exclude librocsolver.so.0 \
    --exclude librocfft.so.0 \
    --exclude librocrand.so.1 \
    --exclude librocsparse.so.0 \
    --exclude librocsparse.so.1 \
    --exclude librocm_smi64.so.2 \
    --exclude librocfft-device-0.so.0 \
    --exclude librocfft-device-1.so.0 \
    --exclude librocfft-device-2.so.0 \
    --exclude librocfft-device-3.so.0 \
    --exclude libc10.so \
    --exclude libc10_hip.so \
    --exclude libtorch.so \
    --exclude libtorch_cpu.so \
    --exclude libtorch_hip.so \
    --exclude libtorch_python.so \
    --exclude libtorchaudio_ffmpeg4.so \
    --exclude libtorchaudio_ffmpeg5.so \
    --exclude libtorchaudio_ffmpeg6.so \
    --exclude libtorchaudio.so \
    --exclude libtorchaudio_sox.so \
    --exclude _torchaudio_ffmpeg4.so \
    --exclude _torchaudio_ffmpeg5.so \
    --exclude _torchaudio_ffmpeg6.so \
    --exclude _torchaudio.so \
    --exclude _torchaudio_sox.so \
    --exclude libomp.so \
    --exclude libsox.so \
    --exclude libavcodec.so.58 \
    --exclude libavcodec.so.59 \
    --exclude libavcodec.so.60 \
    --exclude libavdevice.so.58 \
    --exclude libavdevice.so.59 \
    --exclude libavdevice.so.60 \
    --exclude libavfilter.so.7 \
    --exclude libavfilter.so.8 \
    --exclude libavfilter.so.9 \
    --exclude libavformat.so.58 \
    --exclude libavformat.so.59 \
    --exclude libavformat.so.60 \
    --exclude libavutil.so.56 \
    --exclude libavutil.so.57 \
    --exclude libavutil.so.58 \
    --wheel-dir dist/manylinux \
    dist/torchvision-*.whl

echo 'manylinux wheel:'
ls -lh dist/manylinux/torchvision-*.whl
