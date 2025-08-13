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


def detect_colored_cube(rgb, color):
    """Detect colored object in the camera frame. color: 'red', 'green', 'blue'"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    if color == 'red':
        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    elif color == 'green':
        lower = np.array([40, 70, 70])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color == 'blue':
        lower = np.array([100, 150, 0])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2)  # center pixel
    return None


def move_arm_to_position(joint_indices, target_positions, steps=100):
    global robot_id
    """Move arm smoothly to target positions."""
    current_positions = [p.getJointState(robot_id, j)[0] for j in joint_indices]
    for i in range(steps):
        interpolated = [
            current + (target - current) * (i + 1) / steps
            for current, target in zip(current_positions, target_positions)
        ]
        for j, pos in zip(joint_indices, interpolated):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=pos)
        p.stepSimulation()
        time.sleep(1. / 240.)


def move_arm_ik(target_pos, target_orn=None, end_effector_index=7, steps=100):
    """Move arm to a target position (and orientation) using inverse kinematics."""
    if target_orn is None:
        # Default orientation: pointing down
        target_orn = p.getQuaternionFromEuler([0, math.pi, 0])
    joint_poses = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos, target_orn)
    # Only use as many joints as your robot has
    joint_poses = joint_poses[:len(joint_indices)]
    move_arm_to_position(joint_indices, joint_poses, steps=steps)


# Add a global variable to store click position
clicked_pos = None

def mouse_callback(event, x, y, flags, param):
    global clicked_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pos = (x, y)

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_callback)

# Gripper finger joint indices
LEFT_FINGER = 6
RIGHT_FINGER = 8

# =========================
# MAIN SIMULATION
# =========================
if __name__ == "__main__":
    physics_client = p.connect(p.GUI)  # GUI mode
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

    # End effector index (change if needed)
    end_effector_index = 7  # আপনার রোবটের শেষ লিঙ্কের index দিন

    # Pick & place offsets (cube/tray এর উপরে কিছুটা ওপরে ধরবে)
    pick_offset = [0, 0, 0.08]  # কিউবের উপরে 8cm
    tray_offset = [0, 0, 0.08]  # ট্রের উপরে 8cm

    # Load three cubes with different colors
    red_cube_id = p.loadURDF("cube_small.urdf", [0.4, 0, 0.02])
    green_cube_id = p.loadURDF("cube_small.urdf", [0.3, -0.15, 0.02])
    blue_cube_id = p.loadURDF("cube_small.urdf", [0.3, 0.15, 0.02])

    # Color the cubes
    p.changeVisualShape(red_cube_id, -1, rgbaColor=[1, 0, 0, 1])
    p.changeVisualShape(green_cube_id, -1, rgbaColor=[0, 1, 0, 1])
    p.changeVisualShape(blue_cube_id, -1, rgbaColor=[0, 0, 1, 1])

    # Map color to cube id
    color_cube_map = {
        'red': red_cube_id,
        'green': green_cube_id,
        'blue': blue_cube_id
    }

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    last_pick_time = 0
    pick_cooldown = 3  # seconds

    red_detect_count = 0

    # শুধু লাল কিউব লোড করুন
    red_cube_id = p.loadURDF("cube_small.urdf", [0.4, 0, 0.02])
    p.changeVisualShape(red_cube_id, -1, rgbaColor=[1, 0, 0, 1])
    color_cube_map = {'red': red_cube_id}

    pick_place_count = 0

    try:
        while p.isConnected():
            rgb, depth = get_rgbd()
            center = detect_colored_cube(rgb, 'red')

            # Draw detected center
            if center:
                cv2.circle(rgb, center, 8, (0, 0, 255), 2)

            # Show pick-and-place count
            cv2.putText(rgb, f"Pick & Place Count: {pick_place_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if clicked_pos and center:
                dist = np.linalg.norm(np.array(clicked_pos) - np.array(center))
                if dist < 30:
                    print("[INFO] Mouse clicked on red cube, starting pick and place...")

                    cube_id = red_cube_id
                    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                    tray_pos, _ = p.getBasePositionAndOrientation(tray_id)

                    # 1. Move above the cube (8cm above)
                    pick_above = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.08]
                    move_arm_ik(pick_above, end_effector_index=end_effector_index)
                    time.sleep(0.5)

                    # 2. Lower to just above the cube (2.5cm above cube base)
                    pick_at = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.025]
                    move_arm_ik(pick_at, end_effector_index=end_effector_index)
                    time.sleep(0.3)

                    # 3. Close gripper to grasp
                    p.setJointMotorControl2(robot_id, LEFT_FINGER, p.POSITION_CONTROL, targetPosition=0.0)
                    p.setJointMotorControl2(robot_id, RIGHT_FINGER, p.POSITION_CONTROL, targetPosition=0.0)
                    time.sleep(0.5)

                    # 4. Lift up
                    move_arm_ik(pick_above, end_effector_index=end_effector_index)
                    time.sleep(0.5)

                    # 5. Move above tray (8cm above tray)
                    place_above = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.08]
                    move_arm_ik(place_above, end_effector_index=end_effector_index)
                    time.sleep(0.5)

                    # 6. Lower to just above tray (2.5cm above tray base)
                    place_at = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.025]
                    move_arm_ik(place_at, end_effector_index=end_effector_index)
                    time.sleep(0.3)

                    # 7. Open gripper to release
                    p.setJointMotorControl2(robot_id, LEFT_FINGER, p.POSITION_CONTROL, targetPosition=0.04)
                    p.setJointMotorControl2(robot_id, RIGHT_FINGER, p.POSITION_CONTROL, targetPosition=0.04)
                    time.sleep(0.5)

                    # 8. Lift up again
                    move_arm_ik(place_above, end_effector_index=end_effector_index)
                    time.sleep(0.5)

                    clicked_pos = None
                    pick_place_count += 1

            cv2.imshow("Camera", rgb)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                break
            # No need for p.stepSimulation() here, it's in move_arm_to_position

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()
