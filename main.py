#!/usr/bin/env python3
"""
ACC - keep distance at 30 cm filtered version

Improvements over the basic version:
  - 640x480 resolution (band visible from a greater distance),
  - distance filtered with MEDIAN (removes spikes like 943 mm),
  - real speed computed over a FIXED 0.25 s WINDOW (removes spikes > v_max),
  - target HOLD: when the band disappears for 1-4 frames, the robot does not
    brake immediately but keeps its last decision (no more STOP/JADE flicker).
"""

import numpy as np
import cv2
import math
from collections import deque
from time import monotonic
from gpiozero import Motor, RotaryEncoder

SHOW_PREVIEW = True      # False do autostartu (przy uruchamianiu z systemd)

#  SILNIKI
left  = Motor(forward=6,  backward=5,  enable=12, pwm=True)   # kanal A = lewa
right = Motor(forward=26, backward=16, enable=13, pwm=True)   # kanal B = prawa

#  ENKODERY 
enc_left  = RotaryEncoder(17, 27, max_steps=0)
enc_right = RotaryEncoder(23, 22, max_steps=0)

# KOLO / PREDKOSC 
COUNTS_PER_WHEEL_REV = 223
WHEEL_DIAMETER_M     = 0.067
CIRCUMFERENCE_M      = math.pi * WHEEL_DIAMETER_M   # ~0.2105 m
V_MAX_MS             = 0.80                 # zmierzona predkosc max w pliku speed_test.py

SPEED_WINDOW = 0.25      # s - okno pomiaru predkosci rzeczywistej (wygladza skoki)

# KAMERA 
CAM_INDEX = 0
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#  WYKRYWANIE 
lower = np.array([20,  60,  80])
upper = np.array([45, 255, 255])
MIN_AREA = 100          
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def get_roi_bounds(frame, frac_w=0.8, frac_h=0.7):
    h, w = frame.shape[:2]
    roi_w, roi_h = int(w * frac_w), int(h * frac_h)
    cx, cy = w // 2, h // 2
    return (cy - roi_h // 2, cy + roi_h // 2, cx - roi_w // 2, cx + roi_w // 2)

# ODLEGLOSC (mm) 

DIST_CONST = 26400.0 #stała do przeliczenia szerokości paska w pikselach na odległość w mm (skalibrowana w pliku cam_distance.py)

DIST_BUF = deque(maxlen=5)      # bufor do mediany dystansu

# REGULATOR P
TARGET_MM   = 300.0
KP_V        = 0.002
TOLERANCE   = 20.0
MIN_MOVE_MS = 0.08
HOLD_FRAMES = 4          # ile klatek podtrzymac cel po jego zniknieciu

def drive_speed(v_target):
    # ogranicz zadana predkosc do zakresu 0..V_MAX_MS
    v_target = max(0.0, min(V_MAX_MS, v_target))
    if 0 < v_target < MIN_MOVE_MS:
        v_target = MIN_MOVE_MS
    pwm = max(0.0, min(1.0, v_target / V_MAX_MS))
    left.forward(pwm)
    right.forward(pwm)
    return v_target

def stop():
    #zatrzymaj silniki
    left.stop()
    right.stop()

def steps_to_speed(dsteps, dt):
    # zabezpieczenie przed dzieleniem przez zero
    if dt <= 0:
        return 0.0
    # impulsy -> obroty kola -> metry -> m/s
    return (abs(dsteps) / COUNTS_PER_WHEEL_REV) * CIRCUMFERENCE_M / dt

# PETLA
prev_l = enc_left.steps     # poczatkowy stan lewego enkodera
prev_r = enc_right.steps    # poczatkowy stan prawego enkodera
t_speed = monotonic()       # czas ostatniego pomiaru predkosci
v_real = 0.0                # ostatnia zmierzona predkosc rzeczywista

last_dist = None         # ostatni znany dystans (do podtrzymania)
lost = 0                 # licznik klatek bez celu

try:
    print(f"ACC start. Cel = {TARGET_MM:.0f} mm, v_max = {V_MAX_MS} m/s. 'q'/Ctrl+C konczy.")
    while True:
        ok, frame = cap.read()
        if not ok:              # brak klatki - stop
            stop()
            break

        #  PREDKOSC RZECZYWISTA w stałym oknie
        now = monotonic()
        if now - t_speed >= SPEED_WINDOW:   # licz predkosc tylko co SPEED_WINDOW sekund
            dt = now - t_speed              # faktyczny czas okna
            l, r = enc_left.steps, enc_right.steps      # aktualne stany enkoderow
            # srednia predkosc obu stron
            v_real = (steps_to_speed(l - prev_l, dt) +
                      steps_to_speed(r - prev_r, dt)) / 2.0
            prev_l, prev_r, t_speed = l, r, now     # zapamietaj do nastepnego okna

        # WYKRYWANIE OPASKI
        Y1, Y2, X1, X2 = get_roi_bounds(frame)              # granice obszaru zainteresowania
        roi = frame[Y1:Y2, X1:X2]                           # wytnij ROI
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)          # konwersja BGR -> HSV
        mask = cv2.inRange(hsv, lower, upper)               # maska koloru opaski
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)   # usun drobny szum
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # zasklep dziurki
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)   # znajdz kontury

        w_px = 0
        if contours:
            biggest = max(contours, key=cv2.contourArea)    # najwiekszy obszar = opaska
            x, y, w, h = cv2.boundingRect(biggest)          # prostokat otaczajacy
            if w * h > MIN_AREA:                            # tylko jesli wystarczajaco duzy
                w_px = w                                    # szerokosc opaski w pikselach
                if SHOW_PREVIEW:
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # narysuj ramke

        #  USTALENIE DYSTANSU (mediana i podtrzymanie)
        if w_px > 0:
            DIST_BUF.append(DIST_CONST / w_px)     # dodaj nowy pomiar dystansu do bufora
            dist = float(np.median(DIST_BUF))      # mediana z ostatnich pomiarow (bez skokow)
            last_dist = dist                       # zapamietaj jako ostatni znany
            lost = 0                               # cel widoczny -> zeruj licznik zgubien
            target_ok = True
        else:
            lost += 1                              # cel niewidoczny -> zlicz kolejna klatke
            if lost <= HOLD_FRAMES and last_dist is not None:
                dist = last_dist          # podtrzymanie ostatniej wartosci
                target_ok = True
            else:
                dist = 0.0                         # cel zgubiony na dobre
                target_ok = False
                DIST_BUF.clear()                   # wyczysc bufor dystansu

        #  REGULATOR P 
        if not target_ok:
            stop()          # brak celu - stop
            v_set = 0.0
            status = "BRAK CELU -> STOP"
        else:
            error = dist - TARGET_MM               # >0: za daleko, <0: za blisko
            if abs(error) < TOLERANCE:             # w martwej strefie -> stoj
                stop()
                v_set = 0.0
                status = "DYSTANS OK -> STOP"
            elif error > 0:                        # za daleko -> jedz
                v_set = drive_speed(KP_V * error)  # predkosc proporcjonalna do bledu
                status = "JADE" + (" (podtrzym.)" if w_px == 0 else "")
            else:                                  # za blisko -> stop 
                stop()
                v_set = 0.0
                status = "ZA BLISKO -> STOP"

        print(f"dystans={dist:6.0f}mm  v_zadana={v_set:4.2f}  " f"v_rzecz={v_real:4.2f} m/s  {status}")

        if SHOW_PREVIEW:
            cv2.imshow("ACC (q = koniec)", roi)    # podglad obrazu z ramka
            cv2.imshow("Maska", mask)              # podglad maski
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