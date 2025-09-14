# ---- Base with CUDA toolchain (for building any CUDA/C++ extensions) ----
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Noninteractive apt, sane Python defaults
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # Force CUDA extensions if packages attempt to detect CUDA
    FORCE_CUDA=1 \
    # Target common GPU archs; adjust as needed (30 series, 40 series, 50 series, H100/H200, no A100)
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"

# System packages:
# - python3, pip: Python runtime and package manager
# - build-essential, cmake, ninja-build: build toolchain for native/ CUDA exts
# - ffmpeg, colmap, xvfb: what your script installed at runtime
# - git, wget, curl: useful during builds
# - libgl1, libglib2.0-0, libx* : OpenGL/X deps frequently needed by viewers/tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential cmake ninja-build \
    ffmpeg colmap xvfb \
    git wget curl ca-certificates \
    libgl1 libglib2.0-0 libxext6 libxrender1 libsm6 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN python3 -m pip install --upgrade pip

# ---- PyTorch (GPU) ----
# Use the official CUDA wheels index. Default to cu121 wheels which are broadly compatible
# with 12.x drivers. If you need a specific combo, set at build time:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
#   --build-arg TORCH_PACKAGES="torch torchvision torchaudio"
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu121"
ARG TORCH_PACKAGES="torch torchvision"
RUN python3 -m pip install --extra-index-url ${TORCH_INDEX} ${TORCH_PACKAGES}

# (Optional but often helpful) xformers can speed attention ops
RUN python3 -m pip install --extra-index-url ${TORCH_INDEX} xformers

# Project libraries baked in
RUN python3 -m pip install -r requirements.txt

# Ensure dynamic linker cache is updated for any libs
RUN ldconfig

# Workspace convention
WORKDIR /workspace

# Default shell; keep container interactive by default.
ENTRYPOINT ["/bin/bash"]
