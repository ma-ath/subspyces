FROM nvidia/cuda:11.8.0-base-ubuntu22.04

COPY package-list .
COPY requirements.txt .

# Install apt packages
RUN rm -f /etc/apt/sources.list.d/*.list &&\
    apt-get update &&\
    xargs apt-get -y install < package-list &&\
    rm -rf /var/lib/apt/lists/*

# Install python and pip packages
ARG DEBIAN_FRONTEND=noninteractive
RUN add-apt-repository ppa:deadsnakes/ppa &&\
    apt-get update &&\
    apt-get install -y python3.8 python3.8-venv &&\
    python3.8 -m venv .env &&\
    . .env/bin/activate &&\
    pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt

# Create a working directory.
RUN mkdir /mlima
WORKDIR /mlima

# Entrypoint keep container alive
CMD ["tail", "-f", "/dev/null"]