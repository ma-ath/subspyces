ARG REV=
FROM nvcr.io/nvidia/pytorch:$REV

# Add vscode user with same UID and GID as your host system
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN sudo apt-get update && \
    sudo apt-get upgrade -y && \
    sudo -E apt-get install -y --no-install-recommends\
        build-essential git tmux vim python3-tk