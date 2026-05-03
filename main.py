from backend.kinova import BaseApp
import time
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
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
    calc_inverse_kinematics,
    toEE
)
from trajectory_classes import MultiSegmentTrajectoryGenerator, QuinticPolynomial


# Corner April Tags
BR_CORNER_TAG_POSITION = np.array([0,    -0.42, 0])
TR_CORNER_TAG_POSITION = np.array([0.42, -0.42, 0])
TL_CORNER_TAG_POSITION = np.array([0.42, 0,     0])

BR_CORNER_TAG_ID = 7
TR_CORNER_TAG_ID = 6
TL_CORNER_TAG_ID = 4


# Cup April Tags
POUR_TAG_1_POSITION = np.array([0.3, -0.15, 0])
POUR_TAG_2_POSITION = np.array([0.3, -0.30, 0])
FILL_TAG_POSITION = np.array([0.2, -0.40, 0])

POUR_TAG_1_ID = 5
POUR_TAG_2_ID = 3
FILL_TAG_ID = 2

# Cup Offsets
FILL_CUP_OFFSET = np.array([0.10, 0.082, 0])
POUR_CUP_OFFSET = np.array([0.08, 0, -0.015])

POUR_ABOVE_OFFSET = np.array([0, 0, 0.20])
FILL_ABOVE_OFFSET = np.array([0, 0, 0.11])
SMALL_ABOVE_OFFSET = np.array([0, 0, 0.003])

# Sim Offset
SIM_CUP_OFFSET = np.array([0.065, 0, -0.015])
CUP_APPROACH_OFFSET = np.array([-0.025, 0, 0])

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
                                 cameraPosition=[0.2, -0.2, 1.4],
                                 targetPosition=[0.2, -0.199, 0])
            
            # Adjust for sim difference
            self.kinova_robot.close_gripper()
            
            # Create simulator april tags
            self.kinova_robot.base_kinova.create_apriltag(pos=TL_CORNER_TAG_POSITION, id=TL_CORNER_TAG_ID, size=.1016) # Corner tag
            self.kinova_robot.base_kinova.create_apriltag(pos=TR_CORNER_TAG_POSITION, id=TR_CORNER_TAG_ID, size=.1016) # Corner tag
            self.kinova_robot.base_kinova.create_apriltag(pos=BR_CORNER_TAG_POSITION, id=BR_CORNER_TAG_ID, size=.1016) # Corner tag

            self.kinova_robot.base_kinova.create_apriltag(pos=POUR_TAG_1_POSITION, id=POUR_TAG_1_ID, size=.052) # Pour cup 1 tag
            self.kinova_robot.base_kinova.create_apriltag(pos=POUR_TAG_2_POSITION, id=POUR_TAG_2_ID, size=.052) # Pour cup 1 tag
            self.kinova_robot.base_kinova.create_apriltag(pos=FILL_TAG_POSITION, id=FILL_TAG_ID, size=.052) # Fill cup tag
        else:
            print("Realsense Camera Connected")
            self.cam = RealsenseCamera()
            self.cam.calibration()

        self.cam.start()

        rgb, _ = self.cam.get_frames()
        undistorted = self.cam.undistort(rgb)

        # Calibrate camera using all 3 markers (more robust than single-marker)
        self.cam.calibrate_from_marker(marker_positions={
            TL_CORNER_TAG_ID: TL_CORNER_TAG_POSITION,
            TR_CORNER_TAG_ID: TR_CORNER_TAG_POSITION,
            BR_CORNER_TAG_ID: BR_CORNER_TAG_POSITION,
        })

        # Finds all markers world positions
        self.cam.find_all_markers(undistorted, showIDs=False)
        if len(self.cam.world_positions) > 0:
            print("Accessing world_positions:", self.cam.world_positions.keys())
            self.pour_cup_1 = self.cam.world_positions[POUR_TAG_1_ID] + POUR_CUP_OFFSET
            self.pour_cup_2 = self.cam.world_positions[POUR_TAG_2_ID] + POUR_CUP_OFFSET
            self.fill_cup = self.cam.world_positions[FILL_TAG_ID] + FILL_CUP_OFFSET

        # Show cup positions in sim
        if self.sim_cam:
            self.pour_cup_1[2] += 0.07
            pour_cup_1_pos = (POUR_TAG_1_POSITION + POUR_CUP_OFFSET)
            sim_pour_cup_1 = self.kinova_robot.base_kinova.create_cup(pos=(POUR_TAG_1_POSITION+SIM_CUP_OFFSET))

            self.pour_cup_2[2] += 0.07
            pour_cup_2_pos = (POUR_TAG_2_POSITION + POUR_CUP_OFFSET)
            sim_pour_cup_2 = self.kinova_robot.base_kinova.create_cup(pos=(POUR_TAG_2_POSITION+SIM_CUP_OFFSET))

            self.fill_cup[2] += 0.07
            fill_cup_pos = (FILL_TAG_POSITION + POUR_CUP_OFFSET)
            sim_fill_cup = self.kinova_robot.base_kinova.create_cup(pos=(FILL_TAG_POSITION+POUR_CUP_OFFSET))

        
        # Prepare queue
        self.action_steps = []
        self.action_index = 0

        # Set initial state
        self.state = "WAITING"


        print(f"pour cup 1 pos: {self.pour_cup_1}")
        print(f"pour cup 2 pos: {self.pour_cup_2}")
        print(f"fill cup pos: {self.fill_cup}")
        print("|-------------------------|")
        print("        LOOP STARTED       ")
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
            self.tkinter_setup()


    def _add_pour_sequence(self, cup):
        # 1. Move above cup
        ee = toEE(np.append((cup + POUR_ABOVE_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 2. Move above cup
        ee = toEE(np.append((cup + CUP_APPROACH_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 3. down to cup
        ee = toEE(np.append((cup), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 4. close gripper
        if self.sim: self.action_steps.append((self.kinova_robot.open_gripper, (True,)))
        else: self.action_steps.append((self.kinova_robot.close_gripper, (True,)))

        # 5. lift cup
        ee = toEE(np.append((cup + POUR_ABOVE_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 6. move above fill cup
        ee = toEE(np.append((self.fill_cup + FILL_ABOVE_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 7. turn cup over
        if self.sim: self.action_steps.append((self.move_joint, ([0,0,0,0,0,2,0],)))
        else: self.action_steps.append((self.move_joint, ([0,0,0,0,0,2,100],)))

        # 8. move above fill again
        ee = toEE(np.append((self.fill_cup + FILL_ABOVE_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 9. return above cup
        ee = toEE(np.append((cup + POUR_ABOVE_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 10. lower slightly
        ee = toEE(np.append((cup + SMALL_ABOVE_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 11. open gripper
        if self.sim: self.action_steps.append((self.kinova_robot.close_gripper, (True,)))
        else: self.action_steps.append((self.kinova_robot.open_gripper, (True,)))

        # 12. Move above cup
        ee = toEE(np.append((cup + CUP_APPROACH_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

        # 13. retreat
        ee = toEE(np.append((cup + POUR_ABOVE_OFFSET), [0, pi/2, 0]))
        self.action_steps.append((self.move_ee, (ee,)))

    def move_ee(self, ee, T=5.0, nsteps=2, mode="task"):
        """Move to end effector position using a task-space quintic trajectory."""
        # Build waypoints: [x, y, z, rotx, roty, rotz]\
        curr_angles = self.kinova_robot.get_joint_angles()
        curr_ee, _ = calc_forward_kinematics(curr_angles)

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
                step_ee = toEE(traj.X[0:6,0,k])
                q = calc_inverse_kinematics(step_ee, q)
                self.kinova_robot.set_joint_angles(q)
                self.currrent_angles = q
                if self.sim and k > 0: self.kinova_robot.base_kinova.destroy(ball_ids[k-1])

        elif mode == "joint":
            for k in range(traj.X.shape[2]):
                self.kinova_robot.set_joint_angles(traj.X[:, 0, k])
                self.currrent_angles = q
                if self.sim and k > 0: self.kinova_robot.base_kinova.destroy(ball_ids[k-1])

    def move_joint(self, joint_angles):
        new_angles = self.currrent_angles+np.array(joint_angles[0:6])
        self.kinova_robot.set_joint_angles(new_angles, gripper_percentage=joint_angles[6])
        self.currrent_angles = joint_angles[0:6]

    def pour_cup_button(self, root, cups):
        self.action_steps = []
        self.action_index = 0

        for cup in cups:
            self._add_pour_sequence(cup)

        self.state = "ACTING"
        root.destroy()

    def tkinter_setup(self):
        root = tk.Tk()
        root.geometry("800x600")

        from PIL import Image, ImageTk
        bg_img = ImageTk.PhotoImage(Image.open("menu_background.jpg").resize((800, 600)))

        canvas = tk.Canvas(root, width=800, height=600, highlightthickness=0)
        canvas.place(x=0, y=0, relwidth=1, relheight=1)
        canvas.create_image(0, 0, anchor="nw", image=bg_img)
        canvas.image = bg_img

        # Buttons
        cup_1_button = make_button(
            root=root,
            command=lambda *args: self.pour_cup_button(root, [self.pour_cup_1]),
            text="Water",
            width=25,
            height=1,
            wraplength=0
        )
        cup_2_button = make_button(
            root=root,
            command=lambda *args: self.pour_cup_button(root, [self.pour_cup_2]),
            text="Water",
            width=25,
            height=1,
            wraplength=0
        )
        mix_button = make_button(
            root=root,
            command=lambda *args: self.pour_cup_button(root, [self.pour_cup_1, self.pour_cup_2]),
            text="Mixed (so still just water)",
            width=25,
            height=1,
            wraplength=0
        )

        canvas.create_text(400, 295, text="Pour Cup 1", font=("Arial", 8), fill="gray")
        canvas.create_text(400, 355, text="Pour Cup 2", font=("Arial", 8), fill="gray")

        cup_1_button.place(relx=0.5, rely=0.45, anchor="center")
        cup_2_button.place(relx=0.5, rely=0.55, anchor="center")
        mix_button.place(relx=0.5,   rely=0.65, anchor="center")

        root.mainloop()



if __name__ == "__main__":
    final_project = Main(
        simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        final_project.shutdown()
