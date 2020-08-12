import os
import os.path
import sys
import signal
import connexion
import darknet
import urllib.request
import urllib.error
import flask
import tempfile
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

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

def get_image_type(filename):
    img = Image.open(filename)
    image_type = img.format.lower()
    img.close()
    if not (image_type == 'jpeg' or image_type == 'png'): raise Exception("Image has to be JPEG or PNG")
    return image_type

def annotate(filename, threshold):
    detections = detect(filename, threshold)
    img = Image.open(filename)
    drw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r'DejaVuSans.ttf', 16)
    for detection in detections:
        label = detection[0]
        confidence = detection[1]
        bounds = detection[2]

        box_width = int(float(bounds[2]))
        box_height = int(float(bounds[3]))
        box_center_x = int(float(bounds[0]))
        box_center_y = int(float(bounds[1]))
        x_min = int(box_center_x - box_width/2)
        x_max = int(box_center_x + box_width/2)
        y_min = int(box_center_y - box_height/2)
        y_max = int(box_center_y + box_height/2)

        boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
        drw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=boxColor, width=3)
        txt_width, txt_height = drw.textsize(label, font=font)
        drw.rectangle([x_min, y_min-(txt_height+4), x_min+txt_width+4, y_min], fill=boxColor, outline=boxColor, width=3)
        drw.text((x_min+2, y_min-(txt_height+2)), label, (255, 255, 255), font=font)
    img.save(filename)

def detect_from_url(url, threshold):
    try:
        # Use mkstemp to generate unique temporary filename
        fd, filename = tempfile.mkstemp()
        os.close(fd)
        urllib.request.urlretrieve(url, filename)
        image_type = get_image_type(filename)
        os.rename(filename, filename + '.' + image_type)
        filename = filename + '.' + image_type
        res = detect(filename, threshold)
        os.unlink(filename)
        return res
    except urllib.error.HTTPError as err:
        return 'HTTP error', err.code
    except:
        return 'An error occurred', 500

def detect_from_file():
    try:
        file_to_upload = connexion.request.files['image_file']
        threshold = float(connexion.request.form['threshold'])
        # Use mkstemp to generate unique temporary filename
        fd, filename = tempfile.mkstemp()
        os.close(fd)
        file_to_upload.save(filename)
        image_type = get_image_type(filename)
        os.rename(filename, filename + '.' + image_type)
        filename = filename + '.' + image_type
        res = detect(filename, threshold)
        os.unlink(filename)
        return res
    except urllib.error.HTTPError as err:
        return 'HTTP error', err.code
    except:
        return 'An error occurred', 500

def annotate_from_file():
    try:
        file_to_upload = connexion.request.files['image_file']
        threshold = float(connexion.request.form['threshold'])
        fd, filename = tempfile.mkstemp(".image")
        os.close(fd)
        file_to_upload.save(filename)
        image_type = get_image_type(filename)
        os.rename(filename, filename + '.' + image_type)
        filename = filename + '.' + image_type
        annotate(filename, threshold)
        res = flask.send_file(filename)
        os.unlink(filename)
        return res
    except urllib.error.HTTPError as err:
        return 'HTTP error', err.code
    except:
        return 'An error occurred', 500

def annotate_from_url(url, threshold):
    try:
        fd, filename = tempfile.mkstemp()
        os.close(fd)
        urllib.request.urlretrieve(url, filename)
        image_type = get_image_type(filename)
        os.rename(filename, filename + '.' + image_type)
        filename = filename + '.' + image_type
        annotate(filename, threshold)
        res = flask.send_file(filename)
        os.unlink(filename)
        return res
    except urllib.error.HTTPError as err:
        return 'HTTP error', err.code
    except:
        return 'An error occurred', 500

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
# For compatibility we will make the API available both with and without a version basepath
app.add_api('swagger.yaml')
app.add_api('swagger.yaml', base_path='/1.0')

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)
    app.run(port=8080, server='gevent')
