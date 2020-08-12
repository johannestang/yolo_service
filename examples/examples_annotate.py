import os
import requests

# Make sure the YOLO service is running!
yolo_url = 'http://localhost:8080/'
image_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/dog.jpg'

# Download image for later use:
r = requests.get(image_url)
open('dog.jpg', 'wb').write(r.content)

# First, annotate using URL:
r = requests.get(yolo_url + 'annotate?threshold=0.25&url=' + image_url)
if (r.status_code == 200 and 
        (r.headers['Content-type'] == 'image/jpeg' or r.headers['Content-type'] == 'image/png') and 
        int(r.headers['Content-Length']) > 0):
    open('dog_annotate_url.jpg', 'wb').write(r.content)

# Next, run detection on file:
image = {'image_file': open('dog.jpg', 'rb')}
data = {'threshold': '0.25'}
r = requests.post(yolo_url + 'annotate', files=image, data=data)
if (r.status_code == 200 and 
        (r.headers['Content-type'] == 'image/jpeg' or r.headers['Content-type'] == 'image/png') and 
        int(r.headers['Content-Length']) > 0):
    open('dog_annotate_file.jpg', 'wb').write(r.content)
