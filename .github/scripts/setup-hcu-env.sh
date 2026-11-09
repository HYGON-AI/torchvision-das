#!/usr/bin/env bash
set -euo pipefail

python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
[[ -f /opt/dtk/env.sh ]] || { echo 'DTK is missing' >&2; exit 1; }

# pip selects the newest matching local-version wheel for this Python ABI.
python3 -m pip install --no-cache-dir --pre \
    --only-binary torch,triton,flash-attn \
    --index-url http://12.12.12.201:9929/das_nightly/dtk2610/+simple/ \
    --extra-index-url https://mirrors.huaweicloud.com/artifactory/pypi-public/simple \
    --trusted-host 12.12.12.201 \
    'torch==2.13.0' 'triton==3.7.0' 'flash-attn==2.8.4'
python3 -c 'from importlib.metadata import version; assert "+dtk2610" in version("torch"); assert "dtk2610.torch2130" in version("triton"); assert "dtk2610.torch2130" in version("flash-attn")'

curl -fsSLo /tmp/hcu_llvm_installer.sh \
    http://42.228.13.241:18000/ai_cc/Nightly/hcu_llvm_installer.sh
printf 'y\n' | bash /tmp/hcu_llvm_installer.sh --major 2.0.0
