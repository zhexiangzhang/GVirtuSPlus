FROM nvcr.io/nvidia/pytorch:22.01-py3

USER root

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

RUN apt-get update && \
    apt-get install -y build-essential cmake libxmu-dev libxi-dev libgl-dev libosmesa-dev git liblog4cplus-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-x86_64.tar.gz \
    && tar -zxvf cmake-3.24.1-linux-x86_64.tar.gz \
    && ln -sf /cmake-3.24.1-linux-x86_64/bin/* /usr/bin \
    && cmake --version    