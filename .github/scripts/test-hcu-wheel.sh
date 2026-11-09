#!/usr/bin/env bash
set -euo pipefail

export ROCM_PATH=/opt/dtk
set +u
# shellcheck disable=SC1091
source /opt/dtk/env.sh
set -u
export LD_LIBRARY_PATH="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):/usr/local/lib:${LD_LIBRARY_PATH:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
shopt -s nullglob
wheels=("${repo_root}"/dist/manylinux/torchvision-*.whl)
(( ${#wheels[@]} == 1 )) || { echo 'Expected one manylinux wheel' >&2; exit 1; }

python3 -m pip install --no-cache-dir --no-deps --force-reinstall "${wheels[0]}"
python3 -m pip install --no-cache-dir \
    pytest pytest-mock pytest-xdist pytest-timeout pytest-time pytest-rerunfailures \
    scipy==1.13.1 torchdata \
    -i https://mirrors.huaweicloud.com/artifactory/pypi-public/simple

cd /tmp
PYTHONHASHSEED=0 python3 -m pytest -vv "${repo_root}/test/" \
    --ignore="${repo_root}/test/test_prototype_datasets_builtin.py" \
    --ignore="${repo_root}/test/test_prototype_datasets_utils.py" \
    --ignore="${repo_root}/test/test_prototype_models.py" \
    --ignore="${repo_root}/test/test_prototype_transforms.py" \
    --ignore="${repo_root}/test/test_datasets_download.py" \
    -n 4 --dist=worksteal --timeout=1800 --timeout-method=thread -ra
