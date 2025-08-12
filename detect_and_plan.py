import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time
import random

# -----------------------------
# 1. PYBULLET SETUP
# -----------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

# Load plane
p.loadURDF("plane.urdf")

# Load robot arm
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# -----------------------------
# 2. POSITIONS & OBJECTS
# -----------------------------
# Bin positions
red_box_pos = [0.7, 0.3, 0.05]
green_box_pos = [0.7, 0.0, 0.05]
blue_box_pos = [0.7, -0.3, 0.05]

# Spawn bins
red_box_id = p.loadURDF("cube_small.urdf", red_box_pos, globalScaling=3)
green_box_id = p.loadURDF("cube_small.urdf", green_box_pos, globalScaling=3)
blue_box_id = p.loadURDF("cube_small.urdf", blue_box_pos, globalScaling=3)

p.changeVisualShape(red_box_id, -1, rgbaColor=[1, 0, 0, 1])
p.changeVisualShape(green_box_id, -1, rgbaColor=[0, 1, 0, 1])
p.changeVisualShape(blue_box_id, -1, rgbaColor=[0, 0, 1, 1])

# Cube spawn position
cube_pos = [0.4, 0, 0.05]
cube_id = None
holding_cube = False

# Counters
count_red = 0
count_green = 0
count_blue = 0

# -----------------------------
# 3. ARM CONTROL FUNCTIONS
# -----------------------------
def move_arm_to_position(target_pos, steps=50):
    jp = p.calculateInverseKinematics(robot_id, 6, target_pos)
    for i in range(len(jp)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, jp[i])
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(0.01)

def pick_cube(pos):
    above = [pos[0], pos[1], pos[2] + 0.2]
    move_arm_to_position(above)
    move_arm_to_position(pos)
    print("Picked cube!")

def drop_cube(pos):
    above = [pos[0], pos[1], pos[2] + 0.2]
    move_arm_to_position(above)
    move_arm_to_position(pos)
    print("Dropped cube!")

# -----------------------------
# 4. CAMERA SETUP
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # RED mask
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # GREEN mask
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # BLUE mask
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    detected_color = None
    for color, mask in [("red", mask_red), ("green", mask_green), ("blue", mask_blue)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 500:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            detected_color = color
            break

    if detected_color:
        if cube_id is None:
            cube_id = p.loadURDF("cube_small.urdf", cube_pos)
            if detected_color == "red":
                p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
            elif detected_color == "green":
                p.changeVisualShape(cube_id, -1, rgbaColor=[0, 1, 0, 1])
            elif detected_color == "blue":
                p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 1, 1])

        if not holding_cube:
            pick_cube(cube_pos)
            holding_cube = True

        if detected_color == "red":
            drop_cube([red_box_pos[0], red_box_pos[1], red_box_pos[2] + count_red*0.05])
            count_red += 1
        elif detected_color == "green":
            drop_cube([green_box_pos[0], green_box_pos[1], green_box_pos[2] + count_green*0.05])
            count_green += 1
        elif detected_color == "blue":
            drop_cube([blue_box_pos[0], blue_box_pos[1], blue_box_pos[2] + count_blue*0.05])
            count_blue += 1

        cube_id = None
        holding_cube = False

    # Show counts on frame
    cv2.putText(frame, f"Red: {count_red}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Green: {count_green}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Blue: {count_blue}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Color Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
p.disconnect()
