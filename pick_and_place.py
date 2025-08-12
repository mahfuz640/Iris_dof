"""
pick_and_place.py

Simple PyBullet demo:
- Loads plane + IRIS arm URDF.
- Spawns a colored box as the target object (red by default).
- Uses detect_and_plan.CameraSensor to locate object.
- Uses IK to move end effector above object, descend, create a fixed constraint (grasp),
  lift, move to place position, and release.

Run:
    conda activate iris_arm
    python pick_and_place.py
"""

import time
import pybullet as p
import pybullet_data
import os
import numpy as np
from detect_and_plan import CameraSensor, detect_object_color_from_camera

# ------------------- Config -------------------
URDF_PATH = "C:/Users/RnD Lab/Desktop/Iris_dof/iris_arm/robot_arm/urdf/iris_arm.urdf"   # update path if necessary
END_EFFECTOR_LINK_INDEX = 6   # check your URDF and set correct index
SIM_TIMESTEP = 1.0 / 240.0

# Colors BGR for detect module: red-ish
TARGET_COLOR_LOWER = (0, 0, 150)
TARGET_COLOR_UPPER = (100, 100, 255)

# Place position (world coords) - change as you like
PLACE_POS = [0.0, -0.4, 0.2]

# ------------------------------------------------

def spawn_target_box(position=(0.5, 0, 0.05), size=(0.05, 0.05, 0.05), color=[1, 0, 0, 1]):
    """
    Create a simple box (collision + visual) in the world and return its body id.
    Color is RGBA list.
    """
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[s/2 for s in size],
                                          rgbaColor=color)
    collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[s/2 for s in size])
    box_id = p.createMultiBody(baseMass=0.1,
                               baseCollisionShapeIndex=collision_id,
                               baseVisualShapeIndex=visual_shape_id,
                               basePosition=position)
    return box_id


def move_to_joint_positions(robot_id, target_joints, joint_indices, max_iters=240):
    """
    Smoothly move to target joint positions using POSITION_CONTROL.
    """
    for _ in range(max_iters):
        for i, j_idx in enumerate(joint_indices):
            p.setJointMotorControl2(robot_id, j_idx, p.POSITION_CONTROL, target_joints[i], force=200)
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)


def compute_ik_and_move(robot_id, end_eff_idx, target_pos, joint_indices):
    """
    Compute IK and command joints to move end effector to target_pos.
    Returns joint positions used.
    """
    # default orientation: identity quaternion
    quat = p.getQuaternionFromEuler([0, 1.57, 0])  # try pointing down
    joint_poses = p.calculateInverseKinematics(robot_id, end_eff_idx, target_pos, targetOrientation=quat)
    # Keep only needed joint indices
    target = [joint_poses[i] for i in joint_indices]
    move_to_joint_positions(robot_id, target, joint_indices)
    return target


def main():
    # Start PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    p.setTimeStep(SIM_TIMESTEP)

    # Load plane
    p.loadURDF("plane.urdf")

    # Load robot URDF
    if not os.path.exists(URDF_PATH):
        print(f"URDF not found at {URDF_PATH}. Put iris_arm.urdf in the working folder.")
        return

    robot_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0], useFixedBase=True)

    # Build list of controllable joints (revolute/prismatic)
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        jointType = info[2]
        # revolute(0) or prismatic(1) usually
        if jointType in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joint_indices.append(j)
    print("Controllable joints:", joint_indices)

    # Spawn a target box with red color
    box_id = spawn_target_box(position=[0.5, 0.0, 0.025], size=(0.05, 0.05, 0.05), color=[1, 0, 0, 1])

    # Setup camera
    cam = CameraSensor(cam_target_pos=(0.4, 0, 0.1),
                       cam_distance=1.0, yaw=45, pitch=-30,
                       width=640, height=480, fov=60)

    # Wait a little for rendering to settle
    for _ in range(50):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # HOME pose: try to set a comfortable pose first (if you have meaningful default)
    try:
        # if your robot has same number of joints as joint_indices, set mid positions
        current_positions = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        home_positions = current_positions  # keep same or define explicit home
        move_to_joint_positions(robot_id, home_positions, joint_indices, max_iters=120)
    except Exception as e:
        print("Home pose set skipped:", e)

    # Detect object using camera + OpenCV
    print("Detecting object via camera...")
    detection = None
    for attempt in range(100):
        detection = detect_object_color_from_camera(cam,
                                                   color_lower_bgr=TARGET_COLOR_LOWER,
                                                   color_upper_bgr=TARGET_COLOR_UPPER,
                                                   show_debug=False)
        if detection:
            print("Detected:", detection)
            break
        # step sim and wait
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    if detection is None:
        print("Failed to detect object. Exiting.")
        p.disconnect()
        return

    obj_id = detection['object_id']
    obj_pos = list(detection['world_pos'])
    print(f"Object {obj_id} at {obj_pos}")

    # Plan pick: move above object
    approach_offset = [0, 0, 0.12]
    pick_above = [obj_pos[0] + approach_offset[0],
                  obj_pos[1] + approach_offset[1],
                  obj_pos[2] + approach_offset[2]]
    print("Moving above object to", pick_above)
    compute_ik_and_move(robot_id, END_EFFECTOR_LINK_INDEX, pick_above, joint_indices)

    # Descend to grasp
    grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.02]  # slight offset above base
    print("Descending to", grasp_pos)
    compute_ik_and_move(robot_id, END_EFFECTOR_LINK_INDEX, grasp_pos, joint_indices)
    time.sleep(0.2)

    # Create constraint to simulate grasp
    print("Creating fixed constraint (grasp).")
    # get end-effector world pose
    state = p.getLinkState(robot_id, END_EFFECTOR_LINK_INDEX)
    ee_pos = state[0]
    ee_orn = state[1]
    cid = p.createConstraint(parentBodyUniqueId=robot_id,
                             parentLinkIndex=END_EFFECTOR_LINK_INDEX,
                             childBodyUniqueId=obj_id,
                             childLinkIndex=-1,
                             jointType=p.JOINT_FIXED,
                             jointAxis=[0, 0, 0],
                             parentFramePosition=[0, 0, 0],
                             childFramePosition=[0, 0, 0],
                             parentFrameOrientation=[0, 0, 0, 1],
                             childFrameOrientation=[0, 0, 0, 1])
    # Lift with object
    lift_pos = [pick_above[0], pick_above[1], pick_above[2]]
    print("Lifting to", lift_pos)
    compute_ik_and_move(robot_id, END_EFFECTOR_LINK_INDEX, lift_pos, joint_indices)
    time.sleep(0.2)

    # Move to place position
    place_above = [PLACE_POS[0], PLACE_POS[1], PLACE_POS[2] + 0.12]
    print("Moving to place above:", place_above)
    compute_ik_and_move(robot_id, END_EFFECTOR_LINK_INDEX, place_above, joint_indices)
    time.sleep(0.2)

    # Lower to place
    place_down = [PLACE_POS[0], PLACE_POS[1], PLACE_POS[2] + 0.02]
    print("Lowering to place:", place_down)
    compute_ik_and_move(robot_id, END_EFFECTOR_LINK_INDEX, place_down, joint_indices)
    time.sleep(0.2)

    # Release (remove constraint)
    print("Releasing object.")
    p.removeConstraint(cid)

    # Back to safe pose
    print("Returning to safe pose.")
    compute_ik_and_move(robot_id, END_EFFECTOR_LINK_INDEX, [0.4, 0, 0.3], joint_indices)
    print("Pick-and-place done. You can close the simulation window.")

    # keep GUI open for a while
    for _ in range(500):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    p.disconnect()


if __name__ == "__main__":
    main()
