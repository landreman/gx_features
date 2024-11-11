# Based on https://gitlab.com/NERSC/nersc-official-images/-/blob/main/nersc/mpi4py/3.1.4-cpu/Containerfile

FROM docker.io/ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

RUN \
    apt-get update        && \
    apt-get upgrade --yes && \
    apt-get install --yes    \
        build-essential      \
        gfortran             \
        libcurl4             \
        wget                 \
        git \
        vim              &&  \
    apt-get clean all    &&  \
    rm -rf /var/lib/apt/lists/*


#install miniconda 3.8 (req'd for mpi4py with this OS config)
# ENV installer=Miniconda3-py38_4.12.0-Linux-x86_64.sh
ENV installer=Miniconda3-latest-Linux-x86_64.sh

RUN wget https://repo.anaconda.com/miniconda/$installer && \
    /bin/bash $installer -b -p /opt/miniconda3          && \
    rm -rf $installer

ENV PATH=/opt/miniconda3/bin:$PATH

RUN python3 --version

#need to install mpich in the image
ARG mpich=4.0.2
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch ./configure            && \
    make -j 2                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN /sbin/ldconfig

RUN python3 -m pip install \
    mpi4py==3.1.6 \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    pandas \
    xgboost \
    lightgbm \
    tsfresh \
    mlxtend \
    feature_engine \
    memory_profiler

# RUN python3 -m pip install .
RUN python3 -m pip install git+https://github.com/landreman/gx_features.git

#    https://github.com/landreman/gx_features.git
    

