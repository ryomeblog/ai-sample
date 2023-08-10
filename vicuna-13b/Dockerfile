FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    wget \
    vim \
    software-properties-common \
    gcc-11 \
    g++-11

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.10 python3-pip

RUN pip install llama-cpp-python

WORKDIR /usr/src/data
RUN if [ ! -f /usr/src/data/ggml-vic13b-q4_0.bin ]; then wget https://huggingface.co/eachadea/ggml-vicuna-13b-1.1/resolve/main/ggml-vic13b-q4_0.bin; fi

WORKDIR /usr/src/app

COPY . .
