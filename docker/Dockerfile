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

# # install fucking gdal
# RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y && \
#     apt-get update --fix-missing && \
#     apt-get install --no-install-recommends \
#     # python3-numpy gdal-bin python3-gdal libgdal-dev python3-rtree \
#     -y && \
#     apt-get autoremove -y && apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     rm -rf /usr/local/src/*

RUN pip --no-cache-dir install --upgrade \
    rasterio \
    geopandas \
    descartes \
    rio-tiler \
    pystac \
    ConfigArgParse \
    git+https://github.com/bonlime/pytorch-tools.git@dev

# for fast installationf of new packages 
RUN pip --no-cache-dir install --upgrade \
    supermercado

COPY . .
# COPY . /workspace/jewels/
# COPY ./docker/internal_run.sh /internal_run.sh

# CMD ["bash", "/internal_run.sh"]
