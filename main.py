from backend.kinova import BaseApp
import time
import numpy as np
import pyrealsense2 as rs
import cv2 as cv
import tkinter as tk
from math import pi

from camera import RealsenseCamera
from camera import SimCamera

from ui_helpers import make_button
from kinematics_helpers import (
    EndEffector,
    calc_forward_kinematics,
    calc_numerical_ik,
)
from trajectory_classes import MultiSegmentTrajectoryGenerator, QuinticPolynomial

POS_SCALE = 1.25

POUR_CUP_ID = 6
FILL_CUP_ID = 4

FILL_CUP_OFFSET = np.array([-0.25, 0, 0])
POUR_CUP_OFFSET = np.array([-0.25, 0, 0])

ARM_BASE = np.array([0, 0, 0])
HOME_POSITION = np.array(
    [2.5, 5.76, 2.18, 2.44, 4.54, 0.0])  # From example script


class Main(BaseApp):

    def start(self):

        self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)

        if RealsenseCamera.is_connected():
            self.cam = RealsenseCamera(cameraPosition=[0.2, 0.2, 1.4])
            self.cam.calibration()
        else:
            self.cam = SimCamera(kinova=self.kinova_robot,
                                 cameraPosition=[0.2, 0.2, 1.4],
                                 targetPosition=[0.2, 0.199, 0])

        self.cam.start()

        rgb, _ = self.cam.get_frames()
        undistorted = self.cam.undistort(rgb)

        # Calibrate camera using marker (always flat on table)
        success = self.cam.calibrate_from_marker(
            rgb,
            marker_id=7,
            marker_position=np.array([-0.43*POS_SCALE, 0, 0])
        )

        print(f"Positions: {self.cam.position}")

        # Finds all markers world positions
        self.cam.find_all_markers(undistorted)

        if len(self.cam.world_positions) > 0:
            self.pour_cup = self.cam.world_positions[POUR_CUP_ID]
            self.fill_cup = self.cam.world_positions[FILL_CUP_ID]

        print(f"pour cup pos: {self.pour_cup}")
        print(f"fill cup pos: {self.fill_cup}")

        self.action_steps = []
        self.action_index = 0

        self.state = "WAITING"

        ee, _ = calc_forward_kinematics(self.kinova_robot.get_joint_angles())
        # print(f"EE: pos=[{ee.x:.3f}, {ee.y:.3f}, {ee.z:.3f}] rot=[{ee.rotx:.3f}, {ee.roty:.3f}, {ee.rotz:.3f}]")

        print("|-------------------------|")
        print("           LOOP            ")
        print("|-------------------------|")

    def loop(self):
        if self.state == "ACTING":
            if self.action_index < len(self.action_steps):
                func, args = self.action_steps[self.action_index]
                func(*args)
                self.action_index += 1
            else:
                self.state = "WAITING"

        elif self.state == "WAITING":
            root = tk.Tk()

            button = make_button(
                root=root, command=self.basic_pour_button, text="Make Drink")
            button.pack(padx=20, pady=20)
            root.mainloop()

    def basic_pour_button(self, root):
        self.action_steps = []
        self.action_index = 0

        # 1. Move next to cup
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.pour_cup[0], self.pour_cup[1], self.pour_cup[2]
        ee.rotx, ee.roty, ee.rotz = -pi/2, 0, 0

        self.action_steps.append((self.move, (ee,)))

        # 2. Open gripper
        self.action_steps.append((self.kinova_robot.open_gripper, (True,)))

        # 3. Move onto fill cup position
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.fill_cup[0], self.fill_cup[1], self.fill_cup[2]
        ee.rotx, ee.roty, ee.rotz = -pi/2, 0, 0

        self.action_steps.append((self.move, (ee,)))

        # 4. Close gripper
        self.action_steps.append((self.kinova_robot.close_gripper, (True,)))

        # find desired location from aruco marker
        # make two steps before desired location: move backward and move up

        self.state = "ACTING"
        root.destroy()

    def move(self, ee, T=5.0, nsteps=10, mode="joint"):
        """Move to end effector position using a task-space quintic trajectory."""
        curr_angles = np.array(self.kinova_robot.get_joint_angles()) % (2 * np.pi)
        print(f"[MOVE] Target EE: pos=[{ee.x:.3f}, {ee.y:.3f}, {ee.z:.3f}] rot=[{ee.rotx:.3f}, {ee.roty:.3f}, {ee.rotz:.3f}]")

        # Build waypoints: [x, y, z, rotx, roty, rotz]
        curr_ee, _ = calc_forward_kinematics(curr_angles)

        if mode == "task":
            waypoints = np.array([
                [curr_ee.x, curr_ee.y, curr_ee.z, curr_ee.rotx, curr_ee.roty, curr_ee.rotz],
                [ee.x,      ee.y,      ee.z,      ee.rotx,      ee.roty,      ee.rotz],
            ])
        elif mode == "joint":
            target_angles = calc_numerical_ik(ee, curr_angles)
            waypoints = np.array([curr_angles, target_angles])
        else:
            print("Incorrect Mode")
            return

        traj = MultiSegmentTrajectoryGenerator(method=QuinticPolynomial(), mode="joint", ndof=6)
        traj.solve(waypoints, T=T)
        traj.generate(nsteps_per_segment=nsteps)

        # Visualize path
        ball_ids = [self.kinova_robot.base_kinova.create_ball([ee.x, ee.y, ee.z], end=True)]
        for k in range(1, traj.X.shape[2] - 1):
            pos = traj.X[:3, 0, k].tolist()
            ball_ids.append(self.kinova_robot.base_kinova.create_ball(pos))

        # Move through waypoints: IK at each step, warm-started from previous solution
        q = curr_angles.copy()
        if mode == "task":
            for k in range(traj.X.shape[2]):
                step_ee = EndEffector()
                step_ee.x,    step_ee.y,    step_ee.z    = traj.X[0, 0, k], traj.X[1, 0, k], traj.X[2, 0, k]
                step_ee.rotx, step_ee.roty, step_ee.rotz = traj.X[3, 0, k], traj.X[4, 0, k], traj.X[5, 0, k]
                q = calc_numerical_ik(step_ee, q)
                self.kinova_robot.set_joint_angles(q, gripper_percentage=0)
        elif mode == "joint":
            for k in range(traj.X.shape[2]):
                self.kinova_robot.set_joint_angles(traj.X[:, 0, k], gripper_percentage=0)

        # Remove path 
        for ball_id in ball_ids:
            self.kinova_robot.base_kinova.destroy(ball_id)


if __name__ == "__main__":
    final_project = Main(
        simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        final_project.shutdown()
