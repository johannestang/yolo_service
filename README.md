# YOLO Object Detection Service

Dockerized object detection service using [YOLO](https://pjreddie.com/darknet/yolo/) based on [AlexeyAB's darknet fork](https://github.com/AlexeyAB/darknet) and
exposed as a REST API using [connexion](https://github.com/zalando/connexion). For details see [this post](https://johs.me/posts/object-detection-service-yolo-docker/).

## Quick start

Pull the image from [Docker Hub](https://hub.docker.com/r/johannestang/yolo_service) and spin up a container:
```
docker run -d --rm --name yolo_service -p 8080:8080 johannestang/yolo_service:1.0-yolov3_coco 
```
This will expose two endpoints: `detect` which returns the detected classes, and `annotate` which returns a copy of the image annotated with the detections. Use a GET request if you want to provide an URL to the image, or a POST request if you want to upload an image file.

The service provides a user interface at [localhost:8080/ui](http://localhost:8080/ui) where the endpoints can be tested and the details of the input parameters are listed.

## Image variants

You can build the images yourself using the `build-local.sh` script or pull them from [Docker Hub](https://hub.docker.com/r/johannestang/yolo_service).
They come in multiple variants based on different models/data sets and different configurations of the `darknet` library.
The different models are:

1. YOLOv3 trained on the [COCO dataset](http://cocodataset.org) covering 80 classes listed [here](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names). Tag: `yolov3_coco`.
2. YOLOv3 trained on the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html) covering 601 classes listed [here](https://github.com/AlexeyAB/darknet/blob/master/data/openimages.names). Tag: `yolov3_openimages`.
3. [YOLO9000](https://pjreddie.com/publications/yolo9000/) covering more than 9000 classes listed [here](https://github.com/AlexeyAB/darknet/blob/master/cfg/9k.names). Tag: `yolo9000`.
1. [YOLOv4](https://arxiv.org/abs/2004.10934) trained on the [COCO dataset](http://cocodataset.org) covering 80 classes listed [here](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names). Tag: `yolov4_coco`.

The different `darknet` configurations:

1. The base configuration set up to run on a CPU. Tag: `1.0`
2. Compiled using CUDA 10.0 and cudNN in order to utilize a GPU. Tag: `1.0_cuda10.0`.
3. Compiled using CUDA 10.0 and cudNN with Tensor Cores enabled in order to utilize a GPU with Tensor Cores. Tag: `1.0_cuda10.0_tc`.

When using the CUDA images make sure to use Docker version 19.03 (or newer) and have [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed, then the container can be started by running e.g.:
```
docker run -d --rm --name yolo_service -p 8080:8080 --gpus all johannestang/yolo_service:1.0_cuda10.0-yolov3_coco 
```

