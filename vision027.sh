#!/usr/bin/env bash

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ALIYUN_INDEX="http://mirrors.aliyun.com/pypi/simple/"
readonly HUAWEI_INDEX="https://mirrors.huaweicloud.com/artifactory/pypi-public/simple"
readonly INTERNAL_INDEX="${INTERNAL_PIP_INDEX_URL:?INTERNAL_PIP_INDEX_URL is required}"

select_latest_wheel() {
    python3 - "$INTERNAL_INDEX" "$1" "$2" <<'PY'
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen

from packaging.version import Version

base_url, project, required_text = sys.argv[1:]
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
index_url = urljoin(base_url.rstrip("/") + "/", project.replace("_", "-") + "/")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


parser = Links()
with urlopen(index_url) as response:
    parser.feed(response.read().decode())

candidates = []
for href in parser.hrefs:
    filename = unquote(urlsplit(href).path.rsplit("/", 1)[-1])
    if not filename.endswith(".whl"):
        continue
    if f"-{python_tag}-{python_tag}-" not in filename:
        continue
    if required_text not in filename:
        continue
    version = Version(filename.split("-", 2)[1])
    candidates.append((version, urljoin(index_url, href)))

if not candidates:
    raise SystemExit(
        f"no {project} wheel matching {required_text!r} and {python_tag} in {index_url}"
    )
print(max(candidates)[1])
PY
}

export ROCM_PATH="${ROCM_PATH:-/opt/dtk}"
set +u
# shellcheck disable=SC1091
source "${ROCM_PATH}/env.sh"
if [[ -f "${ROCM_PATH}/cuda/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ROCM_PATH}/cuda/env.sh"
fi
set -u

python3 -c 'import os, sys; expected = os.environ["EXPECTED_PYTHON_VERSION"]; actual = f"{sys.version_info.major}.{sys.version_info.minor}"; assert actual == expected, f"expected Python {expected}, got {actual}"'

case "${EXPECTED_PYTHON_VERSION}" in
    3.10)
        readonly NUMPY_VERSION="1.24.4"
        readonly PANDAS_VERSION="2.0.3"
        readonly SCIPY_VERSION="1.10.1"
        ;;
    3.12)
        readonly NUMPY_VERSION="1.26.4"
        readonly PANDAS_VERSION="2.2.3"
        readonly SCIPY_VERSION="1.14.1"
        ;;
    *) echo "unsupported Python version: ${EXPECTED_PYTHON_VERSION}" >&2; exit 1 ;;
esac

python3 -m pip install --no-cache-dir \
    "mkl==2025.3.0" packaging \
    -i "${ALIYUN_INDEX}" --trusted-host mirrors.aliyun.com

MKL_LIBRARY=""
for root in /opt/intel /usr/local/lib /usr/lib64; do
    if [[ -d "${root}" ]]; then
        MKL_LIBRARY="$(find "${root}" -name libmkl_intel_lp64.so.2 -print -quit 2>/dev/null || true)"
        [[ -n "${MKL_LIBRARY}" ]] && break
    fi
done
if [[ -z "${MKL_LIBRARY}" ]]; then
    echo "libmkl_intel_lp64.so.2 not found in the build image" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$(dirname "${MKL_LIBRARY}"):${LD_LIBRARY_PATH:-}"
echo "MKL library directory: $(dirname "${MKL_LIBRARY}")"

TORCH_WHEEL_URL="$(select_latest_wheel torch 'torch-2.12.0+')"
FLASH_ATTN_WHEEL_URL="$(select_latest_wheel flash-attn '.torch2120.')"
echo "Selected Torch wheel: ${TORCH_WHEEL_URL##*/}"
echo "Selected flash-attn wheel: ${FLASH_ATTN_WHEEL_URL##*/}"
python3 -m pip install --no-cache-dir --upgrade --force-reinstall "${TORCH_WHEEL_URL}"
python3 -m pip install --no-cache-dir --no-deps --force-reinstall "${FLASH_ATTN_WHEEL_URL}"
python3 - <<'PY'
import os
import torch

print(f"torch={torch.__version__}")
print(f"torch.hip={torch.version.hip}")
print(f"gpu_available={torch.cuda.is_available()}")
print(f"torch_path={os.path.dirname(torch.__file__)}")
if torch.version.hip is None:
    raise SystemExit("installed PyTorch is not a ROCm/DCU build")
PY

cd "${REPO_ROOT}"
git submodule update --init --recursive

dnf install -y libjpeg-turbo-devel libwebp-devel
python3 -m pip install --no-cache-dir \
    "numpy==${NUMPY_VERSION}" \
    "pandas==${PANDAS_VERSION}" \
    "scipy==${SCIPY_VERSION}" \
    "setuptools==80.9.0" pybind11 wheel ninja auditwheel patchelf \
    --default-timeout 1000 \
    -i "${HUAWEI_INDEX}"

export USE_MKLDNN=0
export MAX_JOBS="${MAX_JOBS:-32}"
export HIPCC_VERBOSE=1
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-${AMDGPU_TARGETS:-}}"
export LD_LIBRARY_PATH="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"

rm -rf "${REPO_ROOT}/build" "${REPO_ROOT}/dist"
mkdir -p "${REPO_ROOT}/dist/manylinux"

USE_GPU_VIDEO_DECODER=1 \
USE_CPU_VIDEO_DECODER=1 \
python3 setup.py bdist_wheel

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

python3 -m pip install --no-deps --force-reinstall dist/manylinux/torchvision-*.whl
python3 -c 'import numpy, pandas, scipy, torchvision; print(f"numpy={numpy.__version__}"); print(f"pandas={pandas.__version__}"); print(f"scipy={scipy.__version__}"); print(f"torchvision={torchvision.__version__}")'

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
