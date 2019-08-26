#!/bin/bash
docker run -dit --rm --name test_yolo -p 9999:8080 johannestang/yolo_service:1.0-yolov3_coco
sleep 5
curl -L https://github.com/AlexeyAB/darknet/raw/master/data/person.jpg -o person.jpg 
curl -X POST -F 'image_file=@person.jpg' -F threshold=0.25 'http://localhost:9999/detect'
curl -X GET 'http://localhost:9999/detect?url=https%3A%2F%2Fgithub.com%2FAlexeyAB%2Fdarknet%2Fraw%2Fmaster%2Fdata%2Fperson.jpg'
docker stop test_yolo
