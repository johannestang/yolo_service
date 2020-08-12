import os
import sys
import signal
import connexion
import urllib.request
import darknet
import tempfile

# Setup handler to catch SIGTERM from Docker
def sigterm_handler(_signo, _stack_frame):
    print('Sigterm caught - closing down')
    sys.exit()

def detect(filename, threshold):
    im = darknet.load_image(bytes(filename, "ascii"), 0, 0)
    r = darknet.detect_image(network, class_names, im, thresh=threshold)
    darknet.free_image(im)
    # Convert confidence from string to float:
    if len(r) > 0:
        for i in range(len(r)):
            r[i] = (r[i][0], float(r[i][1]), r[i][2])
    return r

def detect_from_url(url, threshold):
    try:
        # Use mkstemp to generate unique temporary filename
        fd, image_file = tempfile.mkstemp(".jpg")
        os.close(fd)
        urllib.request.urlretrieve(url, image_file)
    except:
        return 'Error getting/reading file', 500
    try:
        res = detect(image_file, threshold)
        os.unlink(image_file)
    except:
        return 'Error in detection', 500
    return res

def detect_from_file():
    try:
        uploaded_file = connexion.request.files['image_file']
        threshold = float(connexion.request.form['threshold'])
        # Use mkstemp to generate unique temporary filename
        fd, image_file = tempfile.mkstemp(".jpg")
        os.close(fd)
        uploaded_file.save(image_file)
    except:
        return 'Error in getting/reading file', 500
    try:
        res = detect(image_file, threshold)
        os.unlink(image_file)
    except:
        return 'Error in detection', 500
    return res

# Load YOLO model:
configPath = os.environ.get("config_file")
weightPath = os.environ.get("weights_file")
metaPath = os.environ.get("meta_file")

network, class_names, class_colors = darknet.load_network(
    configPath,
    metaPath,
    weightPath,
    batch_size=1
)

# Create API:
app = connexion.App(__name__)
app.add_api('swagger.yaml')

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)
    app.run(port=8080, server='gevent')
