"""
This is a simple test of the motors and encoders. It runs the motors forward at a fixed speed and prints the encoder steps every SAMPLE_DT seconds. Use Ctrl+C to stop the test.
"""
 
from gpiozero import Motor, RotaryEncoder
from time import sleep, monotonic
 
# --- Motors ---
left  = Motor(forward=6,  backward=5,  enable=12, pwm=True)
right = Motor(forward=26, backward=16, enable=13, pwm=True)
 
# --- Encoders ---
enc_left  = RotaryEncoder(17, 27, max_steps=0)
enc_right = RotaryEncoder(23, 22, max_steps=0)
 
TEST_SPEED = 0.40      
SAMPLE_DT  = 0.5       
 
def stop():
    left.stop()
    right.stop()
 
try:
    # Start motors at test speed
    left.forward(TEST_SPEED)
    right.forward(TEST_SPEED)
 
    # Initialize encoder steps and time
    prev_l, prev_r = enc_left.steps, enc_right.steps
    t_prev = monotonic()
 
    while True:
        sleep(SAMPLE_DT)    #Wait one sample period
        now = monotonic()   #Get current time
        dt = now - t_prev   #Calculate time since last sample
        l, r = enc_left.steps, enc_right.steps  #read current encoder steps
        dl, dr = l - prev_l, r - prev_r         #Calculate change in steps since last sample
        print("LEFT:  steps={:>8d}  d={:>6d}   ||   RIGHT:  steps={:>8d}  d={:>6d}".format(l, dl, r, dr))
        prev_l, prev_r, t_prev = l, r, now
 
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stop()
    print("Motors stopped.")