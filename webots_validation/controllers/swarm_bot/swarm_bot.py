"""Low-level wheel controller for Webots swarm validation bots."""

import json
import math
import sys

from controller import Robot


MAX_WHEEL_SPEED = 12.0
TIME_STEP = 64


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


robot = Robot()
robot_name = sys.argv[1] if len(sys.argv) > 1 else robot.getName()
receiver = robot.getDevice("receiver")
receiver.enable(TIME_STEP)

wheels = [robot.getDevice(name) for name in ("wheel1", "wheel2", "wheel3", "wheel4")]
for wheel in wheels:
    wheel.setPosition(math.inf)
    wheel.setVelocity(0.0)

last_command = [0.0, 0.0]

while robot.step(TIME_STEP) != -1:
    while receiver.getQueueLength() > 0:
        payload = receiver.getString()
        receiver.nextPacket()
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if robot_name in message:
            last_command = message[robot_name]

    left = clamp(float(last_command[0]), -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    right = clamp(float(last_command[1]), -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    wheels[0].setVelocity(left)
    wheels[2].setVelocity(left)
    wheels[1].setVelocity(right)
    wheels[3].setVelocity(right)
