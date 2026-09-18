FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive



RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    clang \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# Install Bazel

RUN curl -L \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    -o /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

WORKDIR /opt

RUN git clone https://github.com/deepmind/code_contests.git

WORKDIR /app



ENTRYPOINT ["bash"]