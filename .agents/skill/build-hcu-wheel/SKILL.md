---
name: build-hcu-wheel
description: Build the torchvision-das 0.27 HCU manylinux wheel with the repository build script.
---

# Build HCU wheel

Use Python 3.10 with DTK 26.04 RC4 or DTK 26.10, a matching HCU Torch 2.12 installation, and the corresponding AICC. The script checks for Python, DTK, Torch, and AICC before installing its remaining build dependencies.

```bash
bash build_hcu_wheel.sh
```

The resulting wheel is in `dist/manylinux/`. PR validation and NIGHTLY release use the same build script through `vision027.sh`. The build script does not install DTK, Torch, or AICC; provision those first.
