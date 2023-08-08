FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    wget \
    vim \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10 python3-pip

RUN pip install ctransformers
RUN pip install dataclasses

WORKDIR /usr/src/data
RUN if [ ! -f /usr/src/data/replit-v2-codeinstruct-3b.q4_1.bin ]; then wget https://huggingface.co/abacaj/Replit-v2-CodeInstruct-3B-ggml/resolve/main/replit-v2-codeinstruct-3b.q4_1.bin; fi

WORKDIR /usr/src/app
COPY . .
