FROM nvcr.io/nvidia/pytorch:19.12-py3

# base setup
RUN apt-get update --fix-missing && \
    apt-get install --no-install-recommends \
    libsm6 libxext6 libxrender-dev \
    wget software-properties-common pkg-config build-essential \
    libglu1-mesa -y && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

RUN pip --no-cache-dir install --upgrade \
    rasterio \
    geopandas \
    descartes \
    rio-tiler \
    pystac \
    ConfigArgParse \
    supermercado \
    git+https://github.com/bonlime/pytorch-tools.git@dev

COPY . .