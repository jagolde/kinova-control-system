from backend.kinova import BaseApp
import time
import numpy as np
import math
import pyrealsense2 as rs
import cv2 as cv

from camera import RealsenseCamera
from camera import SimCamera

class Main(BaseApp):

    def start(self):

        ARM_BASE = np.array([0, 0, 0])
        HOME_POSITION = np.array([1.75, 5.76, 2.18, 2.44, 4.54, 0.0]) # From example script

        pour_cup_id = 1
        fill_cup_id = 2

        self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)

        if RealsenseCamera.is_connected():
            cam = RealsenseCamera(cameraPosition=[0.5, 0, 2])
        else:
            cam = SimCamera(kinova=self.kinova_robot, cameraPosition=[0.5, 0, 2])
        
        cam.start()

        rgb, _ = cam.get_frames()

        undistorted = cam.undistort(rgb)
        cam.find_all_markers(undistorted)

        if len(cam.world_positions) > 0:
            pour_cup = cam.world_positions[pour_cup_id]
            pour_cup_offset = np.array([-0.25, 0, 0]) # April tag offset

            fill_cup = cam.world_positions[fill_cup_id]
            fill_cup_offset = np.array([-0.25, 0, 0]) # April tag offset

        state = "WAITING"

    def loop(self):
        # if state == ACTING
        #   do actions until list is complete
        #   Set state to waiting
        # elif state == WAITING
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

        pass


if __name__ == "__main__":
    final_project = Main(
        simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        final_project.shutdown()
