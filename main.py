from backend.kinova import BaseApp
import time
import numpy as np
import pyrealsense2 as rs
import cv2 as cv
import tkinter as tk
import queue

from camera import RealsenseCamera
from camera import SimCamera

from ui_helpers import make_button
from kinematics_helpers import (
    EndEffector,
    calc_forward_kinematics,
    calc_numerical_ik,
)

POUR_CUP_ID = 1
FILL_CUP_ID = 2

FILL_CUP_OFFSET = np.array([-0.25, 0, 0])
POUR_CUP_OFFSET = np.array([-0.25, 0, 0])

ARM_BASE = np.array([0, 0, 0])
HOME_POSITION = np.array([1.75, 5.76, 2.18, 2.44, 4.54, 0.0]) # From example script

class Main(BaseApp):

    def start(self):
        
        self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)

        if RealsenseCamera.is_connected():
            self.cam = RealsenseCamera(cameraPosition=[0.5, 0, 2])
        else:
            self.cam = SimCamera(kinova=self.kinova_robot, cameraPosition=[0.5, 0, 2])
        
        self.cam.start()

        rgb, _ = self.cam.get_frames()
        undistorted = self.cam.undistort(rgb)

        # Calculate camera position from Base april tag
        self.cam.calibrate_from_marker(rgb)

        # Finds all markers world positions
        self.cam.find_all_markers(undistorted)

        if len(self.cam.world_positions) > 0:
            self.pour_cup = self.cam.world_positions[POUR_CUP_ID]
            self.fill_cup = self.cam.world_positions[FILL_CUP_ID]

        self.action_steps = np.array([], dtype=object)
        self.action_index = 0

        self.state = "WAITING"

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

            button = make_button(root=root, command=self.basic_pour_button, text="Make Drink")
            button.pack(padx=20, pady=20)
            root.mainloop()

    def basic_pour_button(self, root):
        self.action_steps = np.array([], dtype=object)
        self.action_index = 0

        # Move next to cup
        ee = EndEffector()
        ee.x, ee.y, ee.z, ee.rotx, ee.roty, ee.rotz = [0,0,0,0,0,0]
        self.action_steps = np.append(self.action_steps, [(self.move, (ee,))])

        # Open gripper
        self.action_steps = np.append(self.action_steps, [(self.kinova_robot.open_gripper, (True,))])

        # Move onto cup position
        ee = EndEffector()
        ee.x, ee.y, ee.z, ee.rotx, ee.roty, ee.rotz = [0,0,0,0,0,0]
        self.action_steps = np.append(self.action_steps, [(self.move, (ee,))])

        # Close gripper
        self.action_steps = np.append(self.action_steps, [(self.kinova_robot.close_gripper, (True,))])

        # find desired location from aruco marker
        # make two steps before desired location: move backward and move up

        self.state = "ACTING"
        root.destroy()
    
    def move(self, ee):
        """Move to end effector position using inverse kinematics"""
        curr_angles = self.kinova_robot.get_joint_angles()
        
        # Calculate inverse kinematics for the target end effector position
        next_angles = calc_numerical_ik(ee, curr_angles)
        
        # Move to the calculated joint angles
        self.kinova_robot.set_joint_angles(next_angles, gripper_percentage=0)

if __name__ == "__main__":
    final_project = Main(
        simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        final_project.shutdown()
