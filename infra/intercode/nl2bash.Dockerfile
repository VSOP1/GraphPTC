FROM public.ecr.aws/docker/library/ubuntu:latest

ARG file_system_version=1

RUN apt-get update && \
    apt-get install -y bash python3 psmisc bsdmainutils cron imagemagick dnsutils git tree net-tools iputils-ping coreutils curl cpio jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY docker/bash_scripts/setup_nl2b_fs_${file_system_version}.sh /setup.sh
RUN chmod +x /setup.sh && /setup.sh

COPY docker/docker.gitignore /.gitignore
RUN git config --global user.email "intercode@pnlp.org" && \
    git config --global user.name "intercode" && \
    git init && \
    git add -A && \
    git commit -m "initial commit"

WORKDIR /
