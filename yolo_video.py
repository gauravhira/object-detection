import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#                 help="path to input video")

ap.add_argument("-o", "--output", required=True,
                help="path to output video")

ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')

ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')

ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')

args = ap.parse_args()

# Reading class names from classes text file

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generating different colors for each class

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# reading pre-trained model and config file

net = cv2.dnn.readNet(args.weights, args.config)

# getting output layer names in the architecture


def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1]
                     for i in net.getUnconnectedOutLayers()]

    return (output_layers)


# initialize the video stream, pointer to output video file, and
# frame dimensions
# vs = cv2.VideoCapture(args.input)
vs = cv2.VideoCapture(0)
writer = None
(Width, Height) = (None, None)


# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

 # an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# drawing bounding boxes on detected objects with their class names


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    # getting class labels
    label = str(classes[class_id])

    # setting bounding box color
    color = COLORS[class_id]

    # drawing the rectangle
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    # putting class name at top of box
    cv2.putText(img, label, (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # putting confidence score inside the box
    cv2.putText(img, str(confidence), (x+20, y+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


scale = 0.00392

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # if the frame dimensions are empty, grab them
    if Width is None or Height is None:
        (Height, Width) = frame.shape[:2]

    if cv2.waitKey(1) == ord('q'):
        break

    # create input blob
    # opencv assumes images are in BGR so we swapRB channels to resolve this(done by default anyway)
    blob = cv2.dnn.blobFromImage(
        frame, scale, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # blobFromImage(image, scale, (Width, Height), (mean R, G, B), swapRB, crop)

    # setting input blob for the network
    net.setInput(blob)

    start = time.time()
    outs = net.forward(get_output_layers(net))
    end = time.time()

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

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

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)

    # going through the detections remaining after nms and draw bounding boxes

    for i in indices:
        # i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(frame, class_ids[i], confidences[i], round(
            x), round(y), round(x+w), round(y+h))

    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
