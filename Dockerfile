FROM nvidia/cuda:10.0-devel-ubuntu18.04
#FROM ubuntu:18.04

WORKDIR /opt/docker
ENV LD_LIBRARY_PATH=/opt/docker:$LD_LIBRARY_PATH
ARG IMAGE_DIRECTORY=demo_images

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    git-core \
    gcc \
    g++ \
    make \
    vim 

RUN pip3 install setuptools wheel virtualenv awscli --upgrade

# get original darknet source
RUN git clone https://github.com/pjreddie/darknet.git

# modify darknet Makefile source. mostly removes printf statements that are nice for debugging, but not essential to prediction.
COPY src/* darknet/src/
COPY Makefile darknet/

# build the modified source code, and copy the static library (libdarknet.a) and shared library (libdarknet.so)
RUN cd darknet && \
    make GPU=1 all && \
    #make && \
    cp libdarknet.* ..

RUN mkdir images classify_results weights args include cfg data

COPY $IMAGE_DIRECTORY/* images/
COPY weights/* weights/
COPY args/* args/
COPY include/* include/
COPY cfg/* cfg/
COPY data/* data/

COPY predict_classifier_multi.c .

RUN gcc -L$(pwd)/darknet -Wall -o predict_classifier_multi predict_classifier_multi.c -ldarknet && \
    chmod 700 predict_classifier_multi

COPY classify_topk .
