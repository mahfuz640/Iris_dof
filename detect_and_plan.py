import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time

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
# Red bin position (open top tray)
red_box_pos = [0.7, 0.3, 0.02]
red_box_id = p.loadURDF("tray/traybox.urdf", red_box_pos, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=1)
p.changeVisualShape(red_box_id, -1, rgbaColor=[1, 0, 0, 1])

# Cube spawn position (already red)
cube_pos = [0.4, 0, 0.05]
cube_id = p.loadURDF("cube_small.urdf", cube_pos)
p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])  # make cube red
holding_cube = False

# Counter
count_red = 0

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

    detected_red = False
    contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 500:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        detected_red = True

    if detected_red and not holding_cube:
        pick_cube(cube_pos)
        holding_cube = True

        drop_cube([red_box_pos[0], red_box_pos[1], red_box_pos[2] + count_red*0.05])
        count_red += 1

        # Respawn a new red cube
        cube_id = p.loadURDF("cube_small.urdf", cube_pos)
        p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
        holding_cube = False

    # Show counts
    cv2.putText(frame, f"Red Count: {count_red}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Red Detection Only", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
p.disconnect()
