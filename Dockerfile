FROM tensorflow/tensorflow:2.5.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install scikit-learn \
    networkx \
    tqdm \
    matplotlib

EXPOSE 8080