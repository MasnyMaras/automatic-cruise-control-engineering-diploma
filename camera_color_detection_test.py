#!/usr/bin/env python3
"""
COLOR DETECTION - yellow-green color using openCV

"""

import numpy as np
import cv2

# Open the camera at 640x480
CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    # frame height

# HSV color range of the band yellow-green
lower = np.array([20,  60,  80])           #lower HSV bound
upper = np.array([45, 255, 255])           #upper HSV bound

MIN_AREA = 100                              #min area of the blob to be detected
# Elliptical structuring element to clean mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def get_roi_bounds(frame, frac_w=0.8, frac_h=0.7):
    # cropped image is the center of the frame, scaled
    # Auto-scales with resolution - no fixed pixel values needed.
    h, w = frame.shape[:2]
    roi_w, roi_h = int(w * frac_w), int(h * frac_h)
    cx, cy = w // 2, h // 2
    return (cy - roi_h // 2, cy + roi_h // 2,
            cx - roi_w // 2, cx + roi_w // 2)

try:
    print("Color detection. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()             # grab one frame
        if not ok:                         # stop if the camera fails
            break

        # Crop the center of the frame ROI
        Y1, Y2, X1, X2 = get_roi_bounds(frame)
        roi = frame[Y1:Y2, X1:X2]

        #  Build the color mask in HSV 
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)         # BGR to HSV
        mask = cv2.inRange(hsv, lower, upper)              # white where color matches
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)   # remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes

        # Find and pick biggest blob
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)  # largest blob
            x, y, w, h = cv2.boundingRect(biggest)         # blob bounding box
            if w * h > MIN_AREA:                           # only if big enough
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw box
                print(f"w={w:3d} h={h:3d} area={w*h:6d}")                   # print size


        # show image
        cv2.imshow("Image + bbox (q = quit)", roi)
        cv2.imshow("Mask", mask)
        if (cv2.waitKey(1) & 0xFF) == ord("q"): #q to quit
            break
finally:
    cap.release()        
    cv2.destroyAllWindows() 