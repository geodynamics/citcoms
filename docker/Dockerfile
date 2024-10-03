FROM ubuntu:jammy

RUN apt-get update && \
  DEBIAN_FRONTEND='noninteractive' \
  DEBCONF_NONINTERACTIVE_SEEN='true' \
  apt-get install --yes \
    build-essential \
    python-dev-is-python3 \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    git \
    autoconf \
    automake \
    autoconf-archive


RUN useradd \
--create-home \
citcoms_user 

USER citcoms_user

WORKDIR /home/citcoms_user

RUN git clone --recurse-submodules 'https://github.com/geodynamics/citcoms.git'

WORKDIR /home/citcoms_user/citcoms

RUN mkdir aux-config; autoreconf || true; automake --add-missing; autoreconf -i || true; ./configure; make;

ENV PATH="/home/citcoms_user/citcoms/bin:${PATH}"
