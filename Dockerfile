FROM ubuntu:22.04

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python-is-python3 python3-pip  \
        libxkbcommon-x11-0 \
        x11-utils \
        libyaml-dev \
        libegl1-mesa \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libdbus-1-3 \
        libopengl0 \
        gnu && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY . $HOME/src/
WORKDIR $HOME/src/

RUN python -m pip install -r requirements-test.txt

RUN pytest
