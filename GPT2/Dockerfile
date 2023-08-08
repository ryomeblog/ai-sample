FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    wget \
    vim \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10 python3-pip

RUN pip install transformers

WORKDIR /usr/src/data
RUN if [ ! -f /usr/src/data/ggml-model.bin ]; then wget https://huggingface.co/marella/gpt-2-ggml/resolve/main/ggml-model.bin; fi

WORKDIR /usr/src/app
COPY . .
