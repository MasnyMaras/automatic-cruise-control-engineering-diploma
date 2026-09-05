"""
DISTANCE CALIBRATION - measurement at 10 cm (100 mm)
Place the band EXACTLY 100 mm from the camera lens, facing it
(55 mm side horizontal), then run this script.
 
The script:
  - detects the colored band (same HSV range as ACC),
  - measures the bounding box width 'w' in pixels,
  - averages many frames for a stable result,
  - on 'z' prints the ready-to-use constant DIST_CONST.
 
Keys:
  z  -> save the averaged reading and compute the constant
  q  -> quit
"""
 
import numpy as np
import cv2
 
# Calibration distance
D_KAL = 100.0        # mm
 
#  Camera settings
CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     # frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)    # frame height
 
#Color thresholds
lower = np.array([20,  60,  80])           # lower HSV bound
upper = np.array([45, 255, 255])           # upper HSV bound
 
MIN_AREA = 80                               # ignore blobs smaller than this
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # cleaning brush
 
def get_roi_bounds(frame, roi_w=260, roi_h=180):
    # Centered region of interest (ROI)
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    return (cy - roi_h // 2, cy + roi_h // 2, cx - roi_w // 2, cx + roi_w // 2)
 

samples = []
MAX_SAMPLES = 30     # how many recent frames to average
 
print("Place the band 100 mm from the camera. 'z' = save, 'q' = quit.")
 
try:
    while True:
        ok, frame = cap.read()             # grab one frame
        if not ok:                         # stop if the camera fails
            break
 
        # Crop the ROI and build the color mask
        Y1, Y2, X1, X2 = get_roi_bounds(frame)
        roi = frame[Y1:Y2, X1:X2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)         # BGR -> HSV
        mask = cv2.inRange(hsv, lower, upper)              # color mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)   # remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill holes
 
        # Find the biggest blob and take its width
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w_px = 0
        if contours:
            biggest = max(contours, key=cv2.contourArea)  # largest blob
            x, y, w, h = cv2.boundingRect(biggest)         # its bounding box
            if w * h > MIN_AREA:                           # only if big enough
                w_px = w
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw box
 
                samples.append(w_px)                       # add to the average
                if len(samples) > MAX_SAMPLES:             # keep only the last N
                    samples.pop(0)
 
        # Current running average of the width
        avg = sum(samples) / len(samples) if samples else 0
        print(f"w={w_px:3d} px   average({len(samples)})={avg:6.1f} px")
 
        cv2.imshow("Calibration 10cm (z=save, q=quit)", roi)
        cv2.imshow("Mask", mask)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):                # quit
            break
        elif key == ord("z"):              # save + compute the constant
            if avg > 0:
                dist_const = D_KAL * avg   # DIST_CONST = distance * width
                print("\n" + "=" * 50)
                print(f"  Average width at {D_KAL:.0f} mm : {avg:.1f} px")
                print(f"  CONSTANT DIST_CONST = {dist_const:.0f}")
                print(f"  Formula in ACC:  dist_mm = {dist_const:.0f} / w")
                print("=" * 50 + "\n")
            else:
                print("No band detected - nothing saved.")
finally:
    cap.release()              # release the camera
    cv2.destroyAllWindows()    # close all windows