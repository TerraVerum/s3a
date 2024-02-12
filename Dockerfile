FROM ubuntu:22.04

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python-is-python3 python3-pip libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

USER 1001
