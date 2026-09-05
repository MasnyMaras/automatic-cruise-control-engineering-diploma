import numpy as np
import cv2

CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

lower = np.array([20,  90, 110])
upper = np.array([40, 200, 210])

Y1, Y2, X1, X2 = 80, 160, 105, 215
MIN_AREA = 300

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        roi = frame[Y1:Y2, X1:X2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            biggest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(biggest)
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(roi, (cx, cy), 3, (0, 0, 255), -1)
                print(f"bbox w={w} h={h} pole={w*h}  srodek_x={cx}")

        hh, ww = hsv.shape[:2]
        print("HSV srodek ROI:", hsv[hh // 2, ww // 2])

        cv2.imshow("Kolor + bbox (q = koniec)", roi)
        cv2.imshow("Maska HSV", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
