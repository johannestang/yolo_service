import os
import requests

# Make sure the YOLO service is running!
yolo_url = 'http://localhost:8080/'
image_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/dog.jpg'

# Download image for later use:
r = requests.get(image_url)
open('dog.jpg', 'wb').write(r.content)

# First, do detection using URL:
r = requests.get(yolo_url + 'detect?threshold=0.25&url=' + image_url)
if r.status_code == 200:
    print(r.json())

# Next, run detection on file:
image = {'image_file': open('dog.jpg', 'rb')}
data = {'threshold': '0.25'}
r = requests.post(yolo_url + 'detect', files=image, data=data)
if r.status_code == 200:
    print(r.json())
