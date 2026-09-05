"""
ACCELERATION TEST - speed vs power (PWM)
The robot steps power from 10% to 100% and measures speed at each level.
 
Wheel: 67 mm diameter -> ~0.2105 m circumference
Stop: ANY KEY (no Enter needed).

"""
 
import sys
import math
import termios
import tty
import select
from time import sleep, monotonic
from gpiozero import RotaryEncoder, Motor
 
#  WHEEL
WHEEL_DIAMETER_M = 0.067
CIRCUMFERENCE_M  = math.pi * WHEEL_DIAMETER_M      # ~0.2105 m
 
# ENCODER CALIBRATION
COUNTS_PER_WHEEL_REV = 223         # calibrated (mean of 219/225/224)
 
#  ENCODERS 
enc_left  = RotaryEncoder(17, 27, max_steps=0)
enc_right = RotaryEncoder(23, 22, max_steps=0)
 
#  MOTORS 
left  = Motor(forward=6,  backward=5,  enable=12, pwm=True)
right = Motor(forward=26, backward=16, enable=13, pwm=True)
 
#  SETTINGS 
POWER_START = 0.10     # 10%
POWER_STEP  = 0.10     # +10% per step
SETTLE      = 1.5      # s - time to reach steady speed after a power change
MEASURE     = 1.0      # s - speed measurement window
 
 
def key_pressed(): #any key pressed to stop program
    return select.select([sys.stdin], [], [], 0)[0] != []
 
 
def measure_speed(dt_measure):
    # Measure the average speed of both wheels over a time window.
    l0, r0 = enc_left.steps, enc_right.steps        # counts at window start
    t0 = monotonic()
    sleep(dt_measure)                               # wait the measurement window
    dt = monotonic() - t0                           # actual elapsed time
    dl = abs(enc_left.steps - l0)                   # left counts change
    dr = abs(enc_right.steps - r0)                  # right counts change
    # counts -> wheel revolutions -> meters -> m/s
    v_l = (dl / COUNTS_PER_WHEEL_REV) * CIRCUMFERENCE_M / dt
    v_r = (dr / COUNTS_PER_WHEEL_REV) * CIRCUMFERENCE_M / dt
    return (v_l + v_r) / 2.0                         # average of both sides
 
 
def stop():
    left.stop()
    right.stop()
 
 
def wait_settle(seconds):
    #Wait the given time, but abort early if a key is pressed.
    t_end = monotonic() + seconds
    while monotonic() < t_end:
        if key_pressed():
            raise KeyboardInterrupt
        sleep(0.05)
 
 
def main():
    # Switch the terminal so ther is no need to press Enter after a key.
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        print("Acceleration test 10% -> 100%. ANY KEY = STOP.")
        print("-" * 40)
        power = POWER_START
        while power <= 1.0001:                       # step through power levels
            p = min(power, 1.0)
            left.forward(p)                          # set both sides to this power
            right.forward(p)
 
            wait_settle(SETTLE)                      # let it reach steady speed
            v = measure_speed(MEASURE)               # measure the speed
 
            print(f"power={p*100:3.0f}%   v={v:4.2f} m/s   ({v*3.6:4.1f} km/h)")
            power += POWER_STEP                      # next power level
 
        print("-" * 40)
        print("Test finished.")
    except KeyboardInterrupt:
        print("\nStopped by key.")
    finally:
        stop()                                       # always stop the motors
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)   # restore terminal
        print("Motors stopped.")
 
 
if __name__ == "__main__":
    main()