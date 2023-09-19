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
RUN if [ ! -f codellama-34b-instruct.Q2_K.gguf ]; then wget https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/resolve/main/codellama-34b-instruct.Q2_K.gguf; fi

WORKDIR /usr/src/app
COPY . .
