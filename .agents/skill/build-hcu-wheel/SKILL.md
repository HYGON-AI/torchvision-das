---
name: build-hcu-wheel
description: Build the torchvision-das HCU manylinux wheel using the repository build script, including for CI and release work.
---

# Build HCU wheel

Run from the repository root in the DTK 26.10 Rocky Linux 8.6 environment with Python 3.12, HCU torch 2.13, and aicc 2.0 already installed. The script checks DTK, torch, and aicc before installing its other build dependencies.

```bash
bash build_hcu_wheel.sh
```

The finished wheel is in `dist/manylinux/`. Use the same script for local builds, PR CI, and NIGHTLY release. For CI environment preparation and wheel tests, see `.github/scripts/setup-hcu-env.sh` and `.github/scripts/test-hcu-wheel.sh`. Never put the NIGHTLY upload password in the repository; the release workflow reads the `PYPI_PASSWORD` GitHub Actions secret.
