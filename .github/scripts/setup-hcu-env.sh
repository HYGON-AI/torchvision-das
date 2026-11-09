#!/usr/bin/env bash
set -euo pipefail

python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
[[ -f /opt/dtk/env.sh ]] || { echo 'DTK is missing' >&2; exit 1; }
lock_dir="${1:-}"

pip_args=(
    --no-cache-dir --pre
    --only-binary torch,triton,flash-attn
    --index-url http://12.12.12.201:9929/das_nightly/dtk2610/+simple/
    --extra-index-url https://mirrors.huaweicloud.com/artifactory/pypi-public/simple
    --trusted-host 12.12.12.201
)
if [[ -n "${lock_dir}" && -s "${lock_dir}/build-requirements.txt" ]]; then
    python3 -m pip install "${pip_args[@]}" -r "${lock_dir}/build-requirements.txt"
else
    python3 -m pip install "${pip_args[@]}" \
        'torch==2.13.0' 'triton==3.7.0' 'flash-attn==2.8.4'
fi
python3 -c 'from importlib.metadata import version; assert "+dtk2610" in version("torch"); assert "dtk2610.torch2130" in version("triton"); assert "dtk2610.torch2130" in version("flash-attn")'

curl -fsSLo /tmp/hcu_llvm_installer.sh \
    http://42.228.13.241:18000/ai_cc/Nightly/hcu_llvm_installer.sh
if [[ -n "${lock_dir}" && -s "${lock_dir}/aicc-version.txt" ]]; then
    printf 'y\n' | bash /tmp/hcu_llvm_installer.sh --version "$(cat "${lock_dir}/aicc-version.txt")"
else
    printf 'y\n' | bash /tmp/hcu_llvm_installer.sh --major 2.0.0 2>&1 | tee /tmp/aicc-install.log
    if [[ -n "${lock_dir}" ]]; then
        python3 - /tmp/aicc-install.log "${lock_dir}/aicc-version.txt" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
match = re.search(r"完整版本:\s*(\S+)", log)
if not match:
    raise SystemExit("Cannot determine installed aicc version")
Path(sys.argv[2]).write_text(match.group(1) + "\n", encoding="utf-8")
PY
    fi
fi
