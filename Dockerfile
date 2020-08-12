# Define base images
ARG build_image="ubuntu:18.04"
ARG app_image="ubuntu:18.04"

# Build image
FROM ${build_image} AS build
RUN apt-get update
RUN apt-get install -y build-essential git

# Should CUDA be enabled?
ARG cuda=0
# Compile with support for Tensor Cores?
ARG cuda_tc=0

# Get and compile darknet
WORKDIR /src
RUN git clone -n https://github.com/AlexeyAB/darknet.git
WORKDIR /src/darknet
RUN git checkout 38a164bcb9e017f8c9c3645a39419320e217545e
RUN sed -i -e "s!OPENMP=0!OPENMP=1!g" Makefile && \
    sed -i -e "s!AVX=0!AVX=1!g" Makefile && \
    sed -i -e "s!LIBSO=0!LIBSO=1!g" Makefile && \
    sed -i -e "s!GPU=0!GPU=${cuda}!g" Makefile && \
    sed -i -e "s!CUDNN=0!CUDNN=${cuda}!g" Makefile && \
    sed -i -e "s!CUDNN_HALF=0!CUDNN_HALF=${cuda_tc}!g" Makefile && \
    make

# App image:
FROM ${app_image}

# Bare-bones python install
RUN apt-get update && \
    apt-get install -y libgomp1 wget && \
    apt-get install -y --no-install-recommends python3-pip && \
    apt-get install -y python3-setuptools && \
    pip3 install --no-cache-dir wheel && \
    rm -rf /var/lib/apt/lists

# Get darknet from build image
WORKDIR /app
COPY --from=build /src/darknet/libdarknet.so .
COPY --from=build /src/darknet/build/darknet/x64/darknet.py .
COPY --from=build /src/darknet/cfg data/
COPY --from=build /src/darknet/data data/

# Get release version of yolov4.cfg
WORKDIR /app/data
RUN mv yolov4.cfg yolov4.cfg.github
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg
# Reconfigure to avoid out-of-memory errors
RUN sed -i -e "s!subdivisions=8!subdivisions=64!g" yolov4.cfg

# Install api
WORKDIR /app
COPY requirements.txt .
COPY app.py .
COPY swagger.yaml .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model to use (defaults to yolov3_coco):
ARG download_url="https://pjreddie.com/media/files/"
ARG weights_file="yolov3.weights"
ARG config_file="data/yolov3.cfg"
ARG meta_file="data/coco.data"
ENV weights_file=${weights_file}
ENV config_file=${config_file}
ENV meta_file=${meta_file}

# Download trained weights for model:
RUN wget ${download_url}${weights_file}

CMD ["python3", "app.py"]
