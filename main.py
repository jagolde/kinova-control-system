from backend.kinova import BaseApp
import time
import numpy as np
import pyrealsense2 as rs
import cv2 as cv
import tkinter as tk

from camera import RealsenseCamera
from camera import SimCamera

from ux_helpers import make_button
from kinematics_helpers import (
    EndEffector,
    calc_forward_kinematics,
    calc_inverse_kinematics,
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
        self.cam.find_all_markers(undistorted)

        if len(self.cam.world_positions) > 0:
            self.pour_cup = self.cam.world_positions[POUR_CUP_ID]
            self.fill_cup = self.cam.world_positions[FILL_CUP_ID]

        self.state = "WAITING"

    def loop(self):
        if self.state == "ACTING":
            pass
        #   do actions until list is complete
        #   Set state to waiting
        elif self.state == "WAITING":
            root = tk.Tk()

            button = make_button(root=root, command=self.basic_pour_button, text="Make Drink")
            button.pack(padx=20, pady=20)
            root.mainloop()


        #   command input method  --  Button / Screen / Etc...
        #   command -> add list of actions
        #       Go to first cup
        #           Trajectory generation
        #           Movement
        #       Pick up cup
        #       Move up
        #           Trajectory generation
        #           Movement
        #       Go to pour location
        #           Trajectory generation
        #           Movement
        #       Pour
        #       Straighten
        #       Move back
        #           Trajectory generation
        #           Movement
        #       Put down bup
        #   Set state to acting
    
    def basic_pour_button(self, root):
        # Add commands
        self.state = "ACTING"
        root.destroy()

if __name__ == "__main__":
    final_project = Main(
        simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        final_project.shutdown()
