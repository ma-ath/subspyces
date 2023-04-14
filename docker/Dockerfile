FROM nvidia/cuda:11.8.0-base-ubuntu22.04

COPY requirements.txt .

# Install apt packages
RUN apt-get clean && apt-get update &&\
    apt-get -y --no-install-recommends install\
    curl ca-certificates build-essential software-properties-common\
    sudo git vim tmux htop nvtop

# Install python and pip packages
ARG DEBIAN_FRONTEND=noninteractive
RUN add-apt-repository ppa:deadsnakes/ppa &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends python3.8 python3.8-venv &&\
    python3.8 -m venv .env &&\
    . .env/bin/activate &&\
    pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt

# Create a working directory.
RUN mkdir /mlima
WORKDIR /mlima

# Entrypoint keep container alive
CMD ["tail", "-f", "/dev/null"]