# import the necessary packages
import argparse
import datetime
import time
import numpy as np
import cv2


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
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

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("testSTAB.mp4")
bg_frame = None
first_frame = True
bg_dp = 0
display_movement = False

while cap.isOpened():
    (grabbed, frame) = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if bg_frame is not None:
        # compute the absolute difference between the current frame and
        # background frame
        frameDelta = cv2.absdiff(bg_frame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        if display_movement:
            cv2.imshow("FrameDelta", frameDelta)
        cv2.imshow("Thresh", thresh)
        zeroes = np.zeros_like(frameDelta, dtype=np.uint8)
        zeroweight = 0.9
        history = cv2.addWeighted(history, zeroweight, zeroes, 1 - weight, 0)
        cv2.imshow("History", history)
        histweight = 0.9999
        history = cv2.addWeighted(history, histweight, thresh, 1 - weight, 0)

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        histthresh = cv2.threshold(history, 218, 255, cv2.THRESH_BINARY)[1]
        histthresh = cv2.erode(histthresh, None, iterations=5)
        histthresh = cv2.dilate(histthresh, None, iterations=5)
        #cv2.imshow("HistoryT", histthresh)
        (cnts, _) = cv2.findContours(histthresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 100:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
        bboxes = np.array(boxes)
        boxes = non_max_suppression_fast(bboxes, 5)
        for box in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    #time.sleep(0.5)
    key = cv2.waitKey(10) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    # Handle the showing of thresholds and deltas of movemnt
    if key == ord("d"):
        display_movement = not display_movement
        cv2.destroyAllWindows()
    if key == 27:
        break
    if key == ord("b"):
        if first_frame:
            bg_frame = gray
            first_frame = False
        weight = 0.90
        bg_frame = cv2.addWeighted(bg_frame, weight, gray, 1 - weight, 0)
        bg_dp += 1
        history = np.zeros_like(bg_frame, dtype=np.uint8)
        #cv2.imshow("Background", bg_frame)
        print("Data points collected: %s" %bg_dp)

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
