#!/usr/bin/env python3
import numpy as np
import cv2
import math
from time import monotonic
from gpiozero import Motor, RotaryEncoder

SHOW_PREVIEW = True

left  = Motor(forward=6,  backward=5,  enable=12, pwm=True)
right = Motor(forward=26, backward=16, enable=13, pwm=True)

enc_left  = RotaryEncoder(17, 27, max_steps=0)
enc_right = RotaryEncoder(23, 22, max_steps=0)

COUNTS_PER_WHEEL_REV = 223
WHEEL_DIAMETER_M     = 0.067
CIRCUMFERENCE_M      = math.pi * WHEEL_DIAMETER_M
V_MAX_MS             = 0.80

def steps_to_speed(dsteps, dt):
    if dt <= 0:
        return 0.0
    return (abs(dsteps) / COUNTS_PER_WHEEL_REV) * CIRCUMFERENCE_M / dt

CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

lower = np.array([20,  60,  80])
upper = np.array([45, 255, 255])
MIN_AREA = 80
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def get_roi_bounds(frame, roi_w=260, roi_h=180):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    return (cy - roi_h // 2, cy + roi_h // 2,
            cx - roi_w // 2, cx + roi_w // 2)

DIST_CONST = 13200.0

TARGET_MM   = 300.0
KP_V        = 0.002
TOLERANCE   = 20.0
MIN_MOVE_MS = 0.08

def drive_speed(v_target):
    v_target = max(0.0, min(V_MAX_MS, v_target))
    if 0 < v_target < MIN_MOVE_MS:
        v_target = MIN_MOVE_MS
    pwm = v_target / V_MAX_MS
    pwm = max(0.0, min(1.0, pwm))
    left.forward(pwm)
    right.forward(pwm)
    return v_target, pwm

def stop():
    left.stop()
    right.stop()

prev_l = enc_left.steps
prev_r = enc_right.steps
t_prev = monotonic()

try:
    print(f"ACC start. Cel = {TARGET_MM:.0f} mm, v_max = {V_MAX_MS} m/s. 'q'/Ctrl+C konczy.")
    while True:
        ok, frame = cap.read()
        if not ok:
            stop()
            break

        now = monotonic()
        dt = now - t_prev
        l, r = enc_left.steps, enc_right.steps
        v_real = (steps_to_speed(l - prev_l, dt) + steps_to_speed(r - prev_r, dt)) / 2.0
        prev_l, prev_r, t_prev = l, r, now

        Y1, Y2, X1, X2 = get_roi_bounds(frame)
        roi = frame[Y1:Y2, X1:X2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        w_px = 0
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(biggest)
            if w * h > MIN_AREA:
                w_px = w
                if SHOW_PREVIEW:
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if w_px == 0:
            stop()
            v_set, dist = 0.0, 0.0
            status = "BRAK CELU -> STOP"
        else:
            dist = DIST_CONST / w_px
            error = dist - TARGET_MM
            if abs(error) < TOLERANCE:
                stop()
                v_set = 0.0
                status = "DYSTANS OK -> STOP"
            elif error > 0:
                v_set, _ = drive_speed(KP_V * error)
                status = "JADE"
            else:
                stop()
                v_set = 0.0
                status = "ZA BLISKO -> STOP"

        print(f"dystans={dist:6.0f}mm  v_zadana={v_set:4.2f}  "
              f"v_rzecz={v_real:4.2f} m/s  {status}")

        if SHOW_PREVIEW:
            cv2.imshow("ACC (q = koniec)", roi)
            cv2.imshow("Maska", mask)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

except KeyboardInterrupt:
    print("\nPrzerwano.")
finally:
    stop()
    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print("Zatrzymano, kamera zwolniona.")
