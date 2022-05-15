# USAGE
# python object.py

# import the necessary packages

import numpy as np

import imutils
import time
import cv2

from config import *



print("[INFO] running real time object detection.py....")
# initialize the list of class labels MobileNet SSD was trained to

# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

# initialize the video stream, allow the cammera sensor to warmup,
print("[INFO] starting video stream...")

vs = cv2.VideoCapture("uploads/videos/exp5.mp4")
# vs = cv2.VideoCapture(0)

time.sleep(2.0)

detected_objects = []
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels

    ret, frame = vs.read()

    if ret == False:
        break

    frame = imutils.resize(frame, width=800)
    # print(frame.shape)

    cv2.rectangle(frame, pt1=(500, 200), pt2=(750, 400), color=(255, 255, 0), thickness=5)
    roi = frame[200:400, 500:750]

    # grab the frame dimensions and convert it to a blob
    (h, w) = roi.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    person = []
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.50:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if CLASSES[idx] == "person":
                person.append(CLASSES[idx])
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            detected_objects.append(label)
            cv2.rectangle(roi, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(roi, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    if "person" in person:
        cv2.putText(frame, "Alert Person Detected!!!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()

