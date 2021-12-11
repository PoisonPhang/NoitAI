import pyvjoy
import time

control_device = pyvjoy.VJoyDevice(1)

def scale_input(in_raw):
    scaled = in_raw * (16384) + 16384
    if scaled != 0:
        return int (scaled)
    return int (1)

def move_left_stick(x,y):
    #control_device.set_axis(pyvjoy.HID_USAGE_X, scale_input(x))
    #control_device.set_axis(pyvjoy.HID_USAGE_Y, scale_input(y))
    control_device.set_axis(pyvjoy.HID_USAGE_X, 12259)
    control_device.set_axis(pyvjoy.HID_USAGE_Y, 15870)


def move_right_stick(x,y):
    control_device.set_axis(pyvjoy.HID_USAGE_RX, scale_input(x))
    control_device.set_axis(pyvjoy.HID_USAGE_RY, scale_input(y))

def kick():
    control_device.set_button(2, 1)
    time.sleep(1)
    control_device.set_button(2, 0)

def fly():
    control_device.set_button(11, 1)
    time.sleep(5)
    control_device.set_button(11, 0)

def attack():
    control_device.set_button(12, 1)
    time.sleep(0.25)
    control_device.set_button(12, 0)

def switch_right():
    control_device.set_button(6, 1)
    time.sleep(0.25)
    control_device.set_button(6, 0)

kick()
time.sleep(1)
kick()
time.sleep(3)
print('Starting...')


kick()

fly()
attack()

val = -1.0

while val <= 1:
    move_left_stick(val,val)
    move_right_stick(val,val)
    switch_right()
    attack()
    time.sleep(1)
    val += 0.25

move_left_stick(0,0)
move_right_stick(0,0)

kick()

print('Done!')
