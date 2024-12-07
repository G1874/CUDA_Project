FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV RCUTILS_COLORIZED_OUTPUT=1
ARG DEBIAN_FRONTEND=noninteractive

# Basic software
RUN apt update && \
    apt install -y \
    build-essential \
    apt-utils \
    software-properties-common \ 
    unzip \
    curl \
    cmake \
    git 

# Requirements for the Nsight Systems
RUN apt update -y && \
    apt install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    dbus \
    fontconfig \
    gnupg \
    libasound2 \
    libfreetype6 \
    libglib2.0-0 \
    libopengl0 \
    libnss3 \
    libsqlite3-0 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-xkb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxml2 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libxkbfile1 \
    openssh-client \
    wget \
    xcb \
    xkb-data && \
    apt-get clean

# Nsight Systems
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    qtcreator \
    qtbase5-dev \
    qt5-qmake \
    cuda-nsight-systems-12-1 && \
    apt-get clean

# RUN useradd -rm -d /home/developer -s /bin/bash -g root -G sudo -u 1001 developer && \
#     echo 'developer:developer' | chpasswd && \
#     adduser developer sudo
# USER developer

COPY . /home/user/cuda_labs
WORKDIR /home/user/cuda_labs

CMD [ "sleep", "infinity" ]