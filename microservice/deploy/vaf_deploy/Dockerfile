# GPU
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
ADD build.sh /tmp/
ADD servers.tar /
RUN sed -i 's@/archive.ubuntu.com/@/mirrors.aliyun.com/@g' /etc/apt/sources.list && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get clean && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends  git ffmpeg htop tmux unzip vim iotop gcc g++ curl wget fzf zsh fonts-powerline fd-find openmpi-bin libopenmpi-dev mpich perl libc6 libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    chmod +x /tmp/build.sh

RUN /tmp/build.sh
