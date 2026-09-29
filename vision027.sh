#!/usr/bin/env bash

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ALIYUN_INDEX="http://mirrors.aliyun.com/pypi/simple/"
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

if [[ -n "${EXPECTED_PYTHON_VERSION:-}" ]]; then
    python3 -c 'import os, sys; expected = os.environ["EXPECTED_PYTHON_VERSION"]; actual = f"{sys.version_info.major}.{sys.version_info.minor}"; assert actual == expected, f"expected Python {expected}, got {actual}"'
fi

dnf install -y libjpeg-turbo-devel libwebp-devel
python3 -m pip install --no-cache-dir \
    numpy==1.26.4 wheel setuptools ninja \
    -i "${ALIYUN_INDEX}" --trusted-host mirrors.aliyun.com

TORCH_WHEEL_URL="$(select_latest_wheel torch 'torch-2.12.0+')"
FLASH_ATTN_WHEEL_URL="$(select_latest_wheel flash-attn '.torch2120.')"
echo "Selected Torch wheel: ${TORCH_WHEEL_URL##*/}"
echo "Selected flash-attn wheel: ${FLASH_ATTN_WHEEL_URL##*/}"
python3 -m pip install --no-cache-dir --upgrade --force-reinstall "${TORCH_WHEEL_URL}"
python3 -m pip install --no-cache-dir --no-deps --force-reinstall "${FLASH_ATTN_WHEEL_URL}"

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
