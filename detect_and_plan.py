import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import math
import os

# =========================
# USER SETTINGS
# =========================
URDF_DIR = r"C:\Users\Rnd\Desktop\Iris_dof"
ROBOT_URDF = "mybot.urdf"
CUBE_URDF = "cube_small.urdf"

# =========================
# CAMERA SETTINGS
# =========================
CAM_WIDTH, CAM_HEIGHT = 640, 480
CAM_FOV = 60
NEAR, FAR = 0.1, 3.1

def get_rgbd():
    """Capture RGB + Depth image from virtual camera."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1, 0, 1],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 0, 1]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=CAM_FOV,
        aspect=CAM_WIDTH / CAM_HEIGHT,
        nearVal=NEAR,
        farVal=FAR
    )

    width, height, rgb_pixels, depth_pixels, _ = p.getCameraImage(
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.array(rgb_pixels, dtype=np.uint8).reshape((height, width, 4))
    rgb = rgb[:, :, :3]  # Remove alpha
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    depth = np.array(depth_pixels).reshape((height, width))
    depth_m = FAR * NEAR / (FAR - (FAR - NEAR) * depth)

    return rgb, depth_m


def detect_red_cube(rgb):
    """Detect red object in the camera frame."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2)  # center pixel
    return None


def move_arm_to_position(joint_indices, target_positions, steps=100):
    """Move arm smoothly to target positions."""
    current_positions = [p.getJointState(robot_id, j)[0] for j in joint_indices]
    for i in range(steps):
        blend = i / steps
        positions = [
            current_positions[j] + blend * (target_positions[j] - current_positions[j])
            for j in range(len(joint_indices))
        ]
        p.setJointMotorControlArray(robot_id, joint_indices, p.POSITION_CONTROL, positions)
        p.stepSimulation()
        time.sleep(0.01)


# Add a global variable to store click position
clicked_pos = None

def mouse_callback(event, x, y, flags, param):
    global clicked_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pos = (x, y)

# =========================
# MAIN SIMULATION
# =========================
if __name__ == "__main__":
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")
    p.setAdditionalSearchPath(URDF_DIR)

    robot_id = p.loadURDF(os.path.join(URDF_DIR, ROBOT_URDF), useFixedBase=True)
    cube_id = p.loadURDF(CUBE_URDF, [0.4, 0, 0.02])

    # Create red tray using code
    tray_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.02])
    tray_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.1, 0.1, 0.02],
        rgbaColor=[1, 0, 0, 1]  # RED color
    )
    tray_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=tray_collision,
        baseVisualShapeIndex=tray_visual,
        basePosition=[0.6, 0.2, 0.02]
    )

    # Print joint info
    print("=== JOINTS ===")
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        print(f"{j}: joint='{info[1].decode()}', type={info[2]}, link='{info[12].decode()}'")
        if info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(j)

    end_effector_index = 7  # adjust for your robot

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_callback)

    # Simulation loop with safe exit
    try:
        while p.isConnected():
            rgb, depth = get_rgbd()
            center = detect_red_cube(rgb)

            # Draw detected center for visualization
            if center:
                cv2.circle(rgb, center, 8, (0, 255, 0), 2)

            # If user clicked and click is near the cube center, pick and place
            if clicked_pos and center:
                dist = np.linalg.norm(np.array(clicked_pos) - np.array(center))
                if dist < 30:  # 30 pixels tolerance
                    print("[INFO] Mouse clicked on cube, starting pick and place...")
                    move_arm_to_position(joint_indices, [0, 0.5, -0.5, 0, 0, 0])
                    time.sleep(1)
                    p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=0.02)
                    p.setJointMotorControl2(robot_id, 7, p.POSITION_CONTROL, targetPosition=0.02)
                    time.sleep(1)
                    move_arm_to_position(joint_indices, [0, 0, 0, 0.5, 0, 0])
                    time.sleep(1)
                    p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=0.04)
                    p.setJointMotorControl2(robot_id, 7, p.POSITION_CONTROL, targetPosition=0.04)
                    time.sleep(1)
                    clicked_pos = None  # Reset click

            cv2.imshow("Camera", rgb)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Exiting simulation...")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()
