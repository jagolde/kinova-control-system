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
    calc_inverse_kinematics
)
from trajectory_classes import MultiSegmentTrajectoryGenerator, QuinticPolynomial

POS_SCALE = 1

POUR_CUP_ID = 0
FILL_CUP_ID = 1

FILL_CUP_OFFSET = np.array([-0.10, 0, 0.03])
POUR_CUP_OFFSET = np.array([-0.10, 0, 0.03])

ARM_BASE = np.array([0, 0, 0])
HOME_POSITION = np.array(
    [2.5, 5.76, 2.18, 2.44, 4.54, 0.0])  # From example script


class Main(BaseApp):

    def start(self):

        self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)
        self.sim_cam = not RealsenseCamera.is_connected()

        self.currrent_angles = np.array(self.kinova_robot.get_joint_angles())

        if self.sim_cam:
            print("Sim Camera Connected")
            self.cam = SimCamera(kinova=self.kinova_robot,
                                 cameraPosition=[0.2, 0.2, 1.4],
                                 targetPosition=[0.2, 0.199, 0])
        else:
            print("Realsense Camera Connected")
            self.cam = RealsenseCamera(cameraPosition=[0.2, 0.2, 1.4])
            self.cam.calibration()

        self.cam.start()

        rgb, _ = self.cam.get_frames()
        undistorted = self.cam.undistort(rgb)

        # Calibrate camera using all 3 markers (more robust than single-marker)
        if self.sim_cam:
            self.cam.calibrate_from_marker(marker_positions={
                7: [-0.43*POS_SCALE, 0,              0],
                6: [-0.43*POS_SCALE, -0.43*POS_SCALE, 0],
                4: [0,              -0.43*POS_SCALE, 0],
            })
        else:
            self.cam.calibrate_from_marker(marker_positions={
                7: [0,              -0.43*POS_SCALE, 0],
                6: [0.43*POS_SCALE, -0.43*POS_SCALE, 0],
                4: [0.43*POS_SCALE, 0,              0],
            })


        print(f"Position: {self.cam.position}")

        # Finds all markers world positions
        self.cam.find_all_markers(undistorted, showIDs=True, marker_size=0.1016, ids=[4,6,7])
        self.cam.find_all_markers(undistorted, showIDs=True, marker_size=0.508, ids=[0,1,2,3,5])

        if len(self.cam.world_positions) > 0:
            self.pour_cup = self.cam.world_positions[POUR_CUP_ID]
            self.fill_cup = self.cam.world_positions[FILL_CUP_ID]
            self.pour_cup = POUR_CUP_OFFSET
            self.fill_cup = FILL_CUP_OFFSET

        print(f"pour cup pos: {self.pour_cup}")
        print(f"fill cup pos: {self.fill_cup}")

        self.action_steps = []
        self.action_index = 0

        self.state = "WAITING"

        ee, _ = calc_forward_kinematics(self.kinova_robot.get_joint_angles())
        if self.sim: b_id = self.kinova_robot.base_kinova.create_ball([ee.x, ee.y, ee.z], end=True)

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


        # 1. Move above can
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.pour_cup[0]-0.02, self.pour_cup[1], self.pour_cup[2]+0.2
        ee.rotx, ee.roty, ee.rotz = 0, pi/2, 0
        
        self.action_steps.append((self.move_ee, (ee,)))

        # 1. down to cup
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.pour_cup[0]-0.02, self.pour_cup[1], self.pour_cup[2]
        ee.rotx, ee.roty, ee.rotz = 0, pi/2, 0
        
        self.action_steps.append((self.move_ee, (ee,)))

        # 4. Close gripper
        self.action_steps.append((self.kinova_robot.close_gripper, (True,)))

        # 1. lift cup
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.pour_cup[0], self.pour_cup[1], self.pour_cup[2]+0.2
        ee.rotx, ee.roty, ee.rotz = 0, pi/2, 0
        
        self.action_steps.append((self.move_ee, (ee,)))

        # 3. Move above fill cup position
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.fill_cup[0], self.fill_cup[1]+0.05, self.fill_cup[2]+0.17
        ee.rotx, ee.roty, ee.rotz = 0, pi/2, 0
        
        self.action_steps.append((self.move_ee, (ee,)))

        # turn cup over
        bs = [0,0,0,0,0,2,100]
        self.action_steps.append((self.move_joint, (bs,)))

        # 3. unrotate
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.fill_cup[0], self.fill_cup[1], self.fill_cup[2]+0.17
        ee.rotx, ee.roty, ee.rotz = 0, pi/2, 0
        
        self.action_steps.append((self.move_ee, (ee,)))

        # 1. move back
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.pour_cup[0], self.pour_cup[1], self.pour_cup[2]+0.2
        ee.rotx, ee.roty, ee.rotz = 0, pi/2, 0
        
        self.action_steps.append((self.move_ee, (ee,)))

        # 1. move down
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.pour_cup[0], self.pour_cup[1], self.pour_cup[2]+0.01
        ee.rotx, ee.roty, ee.rotz = 0, pi/2, 0
        
        self.action_steps.append((self.move_ee, (ee,)))

        # 4. open gripper
        self.action_steps.append((self.kinova_robot.open_gripper, (True,)))

        # 1. move back
        ee = EndEffector()
        ee.x, ee.y, ee.z = self.pour_cup[0]-0.1, self.pour_cup[1], self.pour_cup[2]+0.2
        ee.rotx, ee.roty, ee.rotz = 0, pi/2, 0
        
        self.action_steps.append((self.move_ee, (ee,)))

        # find desired location from aruco marker
        # make two steps before desired location: move backward and move up

        self.state = "ACTING"
        root.destroy()

    def move_ee(self, ee, T=5.0, nsteps=2, mode="task"):
        """Move to end effector position using a task-space quintic trajectory."""
        # Build waypoints: [x, y, z, rotx, roty, rotz]\
        curr_angles = self.kinova_robot.get_joint_angles()
        curr_ee, _ = calc_forward_kinematics(curr_angles)
        # b_id = self.kinova_robot.base_kinova.create_ball([curr_ee.x, curr_ee.y, curr_ee.z], end=True)

        if mode == "task":
            waypoints = np.array([
                [curr_ee.x, curr_ee.y, curr_ee.z, curr_ee.rotx, curr_ee.roty, curr_ee.rotz],
                [ee.x,      ee.y,      ee.z,      ee.rotx,      ee.roty,      ee.rotz],
            ])
            traj = MultiSegmentTrajectoryGenerator(method=QuinticPolynomial(), mode="task", ndof=6)
        elif mode == "joint":
            target_angles = calc_numerical_ik(ee, curr_angles)
            waypoints = np.array([curr_angles, target_angles])
            traj = MultiSegmentTrajectoryGenerator(method=QuinticPolynomial(), mode="joint", ndof=6)
        else:
            print("Incorrect Mode")
            return

        traj.solve(waypoints, T=T)
        traj.generate(nsteps_per_segment=nsteps)

        # Visualize path
        if self.sim:
            ball_ids = []
            for k in range(1, traj.X.shape[2] - 1):
                pos = traj.X[:3, 0, k].tolist()
                ball_ids.append(self.kinova_robot.base_kinova.create_ball(pos))
            ball_ids.append(self.kinova_robot.base_kinova.create_ball([ee.x, ee.y, ee.z], end=True))

        # Move through waypoints: IK at each step, warm-started from previous solution
        q = curr_angles.copy()
        if mode == "task":
            for k in range(traj.X.shape[2]):
                step_ee = EndEffector()
                step_ee.x,    step_ee.y,    step_ee.z    = traj.X[0, 0, k], traj.X[1, 0, k], traj.X[2, 0, k]
                step_ee.rotx, step_ee.roty, step_ee.rotz = traj.X[3, 0, k], traj.X[4, 0, k], traj.X[5, 0, k]
                q = calc_inverse_kinematics(step_ee, q)
                self.kinova_robot.set_joint_angles(q)
                self.currrent_angles = q
                if self.sim: self.kinova_robot.base_kinova.destroy(ball_ids[k])
            # q = calc_inverse_kinematics(ee, q)
            # self.kinova_robot.set_joint_angles(q, gripper_percentage=0)
        elif mode == "joint":
            for k in range(traj.X.shape[2]):
                self.kinova_robot.set_joint_angles(traj.X[:, 0, k])
                self.currrent_angles = q
                if self.sim: self.kinova_robot.base_kinova.destroy(ball_ids[k])

    def move_joint(self, joint_angles):
        new_angles = self.currrent_angles+np.array(joint_angles[0:6])
        print(f"current: {self.currrent_angles} new: {new_angles}")
        self.kinova_robot.set_joint_angles(new_angles, gripper_percentage=joint_angles[6])
        self.currrent_angles = joint_angles[0:6]


if __name__ == "__main__":
    final_project = Main(
        simulate=False, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        final_project.shutdown()
