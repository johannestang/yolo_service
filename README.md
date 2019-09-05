# YOLO Object Detection Service

Dockerized object detection service using [YOLO](https://pjreddie.com/darknet/yolo/) based on [AlexeyAB's darknet fork](https://github.com/AlexeyAB/darknet) and
exposed as a REST API using [connexion](https://github.com/zalando/connexion). For details see [this post](https://johs.me/posts/object-detection-service-yolo-docker/).

## Quick start

Pull the image from [Docker Hub](https://hub.docker.com/r/johannestang/yolo_service) and spin up a container:
```
docker run -d --rm --name yolo_service -p 8080:8080 johannestang/yolo_service:1.0-yolov3_coco 
```

This will expose a single endpoint `detect` that accepts GET and POST requests where the former takes a URL of an image and the latter lets you upload an image for detection.
The service provides a user interface at [localhost:8080/ui](http://localhost:8080/ui) where the endpoint can be tested and the details of the input parameters are listed.

## Image variants

You can build the images yourself using the `build-local.sh` script or pull them from [Docker Hub](https://hub.docker.com/r/johannestang/yolo_service).
They come in nine variants based on three different models/data sets and three different configurations of the `darknet` library.
The different models are:

1. YOLOv3 trained on the [COCO dataset](http://cocodataset.org) covering 80 classes listed [here](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names). Tag: `yolov3_coco`.
2. YOLOv3 trained on the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html) covering 601 classes listed [here](https://github.com/AlexeyAB/darknet/blob/master/data/openimages.names). Tag: `yolov3_openimages`.
3. [YOLO9000](https://pjreddie.com/publications/yolo9000/) covering more than 9000 classes listed [here](https://github.com/AlexeyAB/darknet/blob/master/cfg/9k.names). Tag: `yolo90000`.

The different `darknet` configurations:

1. The base configuration set up to run on a CPU. Tag: `1.0`
2. Compiled using CUDA 10.0 and cudNN in order to utilize a GPU. Tag: `1.0_cuda10.0`.
3. Compiled using CUDA 10.0 and cudNN with Tensor Cores enabled in order to utilize a GPU with Tensor Cores. Tag: `1.0_cuda10.0_tc`.

When using the CUDA images make sure to use Docker version 19.03 (or newer) and have [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed, then the container can be started by running e.g.:
```
docker run -d --rm --name yolo_service -p 8080:8080 --gpus all johannestang/yolo_service:1.0_cuda10.0-yolov3_coco 
```

