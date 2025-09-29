# syntax=docker/dockerfile:1
FROM vastai/pytorch:2.2.2-cuda-12.1.1-py310-ipv2

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get -o Acquire::http::Pipeline-Depth="0" \
            -o Acquire::http::No-Cache=true \
            -o Acquire::BrokenProxy=true \
    update && \
    apt-get -o Acquire::http::Pipeline-Depth="0" \
            -o Acquire::http::No-Cache=true \
            -o Acquire::BrokenProxy=true \
    install -y --no-install-recommends \
    build-essential cmake git wget curl ca-certificates \
    python3.10 python3-pip python3.10-dev python3.10-venv \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev \
    libeigen3-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev libboost-all-dev \
    ffmpeg vim tmux htop && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get -o Acquire::http::Pipeline-Depth="0" \
            -o Acquire::http::No-Cache=true \
            -o Acquire::BrokenProxy=true \
    update && \
    apt-get -o Acquire::http::Pipeline-Depth="0" \
            -o Acquire::http::No-Cache=true \
            -o Acquire::BrokenProxy=true \
    install -y --no-install-recommends \
    libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev \
    libceres-dev libflann-dev s3cmd || true && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/colmap/colmap.git /tmp/colmap \
    && cd /tmp/colmap \
    && git checkout 3.9.1 \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/colmap


RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
    && tar xzf s5cmd_2.2.2_Linux-64bit.tar.gz \
    && mv s5cmd /usr/local/bin/ \
    && rm s5cmd_2.2.2_Linux-64bit.tar.gz

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install numpy scipy matplotlib opencv-python-headless nerfstudio kornia transformers datasets scipy

WORKDIR /workspace
CMD ["/bin/bash"]