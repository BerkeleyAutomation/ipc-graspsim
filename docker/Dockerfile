# FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
FROM ubuntu:20.04
# env variables for tzdata install
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Vancouver

RUN apt-get update -y && \
    # off-screen rendering
    apt-get install -y wget llvm-6.0 llvm-6.0-tools freeglut3 freeglut3-dev &&\
    wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb && \
    (dpkg -i ./mesa_18.3.3-0.deb || true) && apt install -f && \
    # remaining dependencies
    apt-get install -y \
        python3-pip \ 
        python3-dev \
        ninja-build \
        freeglut3-dev \
	# Remaining dependencies
        liboctomap-dev libfcl-dev \
        gifsicle libfreetype6-dev libpng-dev \
        build-essential libsuitesparse-dev libmetis-dev \
        libgmp-dev libopenblas-dev libomp-dev cmake intel-mkl \
        git libglu1-mesa-dev mesa-common-dev \
        xorg-dev libboost-program-options-dev libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

# Install all other python deps
RUN git clone https://github.com/mmatl/pyopengl.git && python3 -m pip install --no-cache-dir ./pyopengl
RUN python3 -m pip install "ruamel.yaml<0.18.0"
RUN python3 -m pip install --no-cache-dir trimesh python-fcl autolab_core numpy pandas pyrender tqdm matplotlib

RUN mkdir -p IPC_sim/build
COPY ["./CMakeLists.txt", "IPC_sim/CMakeLists.txt"]
COPY ["./cmake", "IPC_sim/cmake"]
COPY ["./tests/", "IPC_sim/tests/"]
COPY ["./src/", "IPC_sim/src/"]
COPY ["./Format/", "IPC_sim/Format/"]
RUN (cd IPC_sim/build && cmake -DCMAKE_BUILD_TYPE=Release ..)

# Install repo
RUN (cd IPC_sim/build && make -j4)

COPY ["./batch.py", "IPC_sim/batch.py"]

WORKDIR /IPC_sim
