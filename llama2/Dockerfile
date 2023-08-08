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
RUN if [ ! -f llama-2-13b-chat.ggmlv3.q6_K.bin ]; then wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q6_K.bin; fi

WORKDIR /usr/src/app
COPY . .
