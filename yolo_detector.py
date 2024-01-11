import cv2
import argparse
import numpy as np

# ------Parsing arguments--------

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True, help='path to inout image')

ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')

ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')

ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')

args = ap.parse_args()

# -------Input----------

image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]

scale = 0.00392


# Reading class names from classes text file

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generating different colors for each class

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# reading pre-trained model and config file

net = cv2.dnn.readNet(args.weights, args.config)

# create input blob

# opencv assumes images are in BGR so we swapRB channels to resolve this(done by default anyway)
blob = cv2.dnn.blobFromImage(
    image, scale, (416, 416), (0, 0, 0), swapRB=True, crop=False)
# blobFromImage(image, scale, (Width, Height), (mean R, G, B), swapRB, crop)

# setting input blob for the network
net.setInput(blob)

# getting output layer names in the architecture


def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1]
                     for i in net.getUnconnectedOutLayers()]

    return (output_layers)

# drawing bounding boxes on detected objects with their class names


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(img, str(confidence), (x+20, y+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# run inference through network and get predictions from output layers

outs = net.forward(get_output_layers(net))

# initializing prediction variables

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# from each detection from each output layer get
# confidence, class id, bounding box parameters
# ignore weak detections (confidence < 0.5)

for out in outs:
    for detection in out:
        # gets the confidence scores of classes which match the object for a particular bounding box
        scores = detection[5:]
        # returns index of class id which has the max confidence (matches the most)
        class_id = np.argmax(scores)
        # store the confidence value present at index class_id from scores
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0]*Width)
            center_y = int(detection[1]*Height)
            w = int(detection[2]*Width)
            h = int(detection[3]*Height)

            x = center_x - w / 2
            y = center_y - h / 2

            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append([x, y, w, h])

# non max suppression (removes overlapping bounding boxes)

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# going through the detections remaining after nms and draw bounding boxes

for i in indices:
    # i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    draw_bounding_box(image, class_ids[i], confidences[i], round(
        x), round(y), round(x+w), round(y+h))

# display output image
cv2.imshow("object detection", image)

# waiting until any key is pressed
cv2.waitKey()

# saving output image
cv2.imwrite("object-detection.jpg", image)

# release resources
cv2.destroyAllWindows()
