import numpy as np
import cv2
from datetime import datetime


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    print("No of boxes:%s" %len(boxes))
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

cap = cv2.VideoCapture("C:\TestFootage\DropAndCollect360.mp4")
bg = None
while cap.isOpened():
    # Capture a frame then blur ir and make it grey to reduce required process power
    (grabbed, frame) = cap.read()
    frame = cv2.GaussianBlur(frame, (21, 21), 0)
    if bg is None:
        bg = frame

    frameDelta = cv2.absdiff(bg, frame)
    # Create a threshold
    thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
    b, g, r = cv2.split(thresh)
    thresh = cv2.bitwise_or(b,g)
    thresh = cv2.bitwise_or(thresh, r)
    thresh = cv2.erode(thresh, None, iterations=7)
    thresh = cv2.dilate(thresh, None, iterations=9)

    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    i = 0
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 200:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        ar = float(h)/float(w)
        cv2.putText(frame, "[%s] Aspect Ratio: %.2f / Area = %s" %(i, ar, cv2.contourArea(c)), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        if ar > 1.5:
            cv2.putText(frame, "Human", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        if 1.5 > ar:
            cv2.putText(frame, "Bag", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        boxes.append((x, y, w, h))
        i+=1
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    boxes = np.array(boxes)
    supboxes = non_max_suppression_fast(boxes, 0.3)
    print("new No of boxes:%s" % len(supboxes))
    for (x, y, w, h) in supboxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imshow("FrameDelta", frameDelta)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame", frame)
    #time.sleep(0.5)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    if key == 27:
        break
# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()