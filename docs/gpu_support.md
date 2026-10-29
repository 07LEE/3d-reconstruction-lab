---
title: GPU, CUDA and PyTorch Support
description: Verified hardware configuration, supported GPU generations, CUDA toolkit and driver requirements, and how to re-check them.
category: guide
last_updated: 2026-10-29
---

# GPU, CUDA and PyTorch Support

## Requirements

- NVIDIA GPU with 16GB+ VRAM recommended
- CUDA Toolkit 13.2 for the unified `gs` environment, created as the `cuda132` Conda environment by `00_setup_environment.sh`
- PyTorch 2.11.0+cu130; native `sm_120` CUBIN is required for Blackwell
- NVIDIA driver whose `nvidia-smi` header reports CUDA 13.2 or newer (tested with 595.99)

A CUDA toolkit is only needed to build the extensions (rasterizers, tetranerf, pytorch3d) and for nvdiffrast, which compiles during training. Its CUDA major version must match the PyTorch build. Running prebuilt PyTorch needs only the driver.

## Verified Configuration

- GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell Architecture, `sm_120`, 16GB VRAM)
- CUDA: CUDA 13.2 Toolkit / PyTorch 2.11.0+cu130
- Host OS: Ubuntu 26.04 LTS, glibc 2.43 (GCC 12 host compiler via `-ccbin /usr/bin/g++-12`); CUDA 12.x headers do not compile against this glibc, see [Blackwell Build Notes](blackwell_build_notes.md)
- CUBIN Binaries: Native `sm_120` SASS CUBIN with `12.0+PTX` fallback
- Rasterizer Performance Benchmark (RTX 5070 Ti, 1,112-frame indoor scene, measured with CUDA 12.8):
  - PTX JIT Fallback (`sm_90` SASS + JIT on `sm_120`): `~37.18 it/s`
  - Native `sm_120` SASS CUBIN: `~66.09 it/s` (`~1.78x` throughput performance gain)

## GPU Support

Only an RTX 5070 Ti (`sm_120`) has been tested. The table lists what the installed tools report. The extensions are built for `TORCH_CUDA_ARCH_LIST` (default `12.0+PTX`), so on another GPU set it to that GPU's compute capability before building, for example `TORCH_CUDA_ARCH_LIST=8.9 ./scripts/00_setup_environment.sh --env gs`.

| GPU generation | Compute capability | nvcc 13.2 | nvcc 12.8 (previous setup) | PyTorch 2.11.0 prebuilt kernels |
| --- | --- | --- | --- | --- |
| Maxwell, Pascal, Volta (GTX 9xx/10xx, V100) | 5.x, 6.x, 7.0 | no | yes | no |
| Turing (RTX 20xx, GTX 16xx) | 7.5 | yes | yes | yes |
| Ampere (RTX 30xx) | 8.0, 8.6 | yes | yes | yes |
| Ada (RTX 40xx) | 8.9 | yes | yes | yes |
| Hopper, Blackwell (H100, B200, RTX 50xx) | 9.0, 10.0, 12.0 | yes | yes | yes |

Re-check after a toolkit or PyTorch upgrade (last checked 2026-10-29):

```bash
$(conda info --base)/envs/cuda132/bin/nvcc --list-gpu-arch      # architectures the toolkit can build
python -c "import torch; print(torch.cuda.get_arch_list())"      # architectures PyTorch ships kernels for
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

Related: [Blackwell Build Notes](blackwell_build_notes.md), [ADR 0020](adr/0020-unified-gs-environment-on-cuda-13-2.md)
