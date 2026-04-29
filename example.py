from backend.kinova import BaseApp
import numpy as np
from kinematics_helpers import *
import time

class Main(BaseApp):
        
    def start(self):
        self.home = False

        ee = EndEffector()
        # above cup
        ee.x = 0.25
        ee.y = -0.35
        ee.z = 0.25
        ee.rotx = 0
        ee.roty = pi/2
        ee.rotz = 0
        self.angles = calc_inverse_kinematics(ee,joint_values)
        # down to cup
        ee.x = 0.25
        ee.y = -0.35
        ee.z = 0.025
        ee.rotx = 0
        ee.roty = pi/2
        ee.rotz = 0
        self.angles1 = calc_inverse_kinematics(ee,joint_values)
        # move in to cup
        ee.x = 0.35
        ee.y = -0.35
        ee.z = 0.025
        ee.rotx = 0
        ee.roty = pi/2
        ee.rotz = 0
        self.angles2 = calc_inverse_kinematics(ee,joint_values)
        # lift can over cup
        ee.x = 0.2
        ee.y = -0.1
        ee.z = 0.2
        ee.rotx = 0
        ee.roty = pi/2
        ee.rotz = 0
        self.angles3 = calc_inverse_kinematics(ee,joint_values)
        # rotate cup
        self.angles4 = self.angles3.copy()
        if self.angles4[5] + 2.5 > 2.6:
            self.angles4[5] -= 2.5
        else:
            self.angles4[5] += 2.5
        
    def loop(self):
        #HOME_POSITION = np.array([1.75, 5.76, 2.18, 2.44, 4.54, 0.0])
        HOME_POSITION = np.array([0, 0, 0, 0, 0, 0])
        next_position = np.array([0.79, 6.11, 1.48, 1.4, 6.11, 1.57])
        if(self.home):
            self.kinova_robot.set_joint_angles(self.angles, gripper_percentage=0)
            time.sleep(1)
            self.kinova_robot.set_joint_angles(self.angles1, gripper_percentage=0)
            time.sleep(1)
            self.kinova_robot.set_joint_angles(self.angles2, gripper_percentage=60)
            time.sleep(1)
            self.kinova_robot.set_joint_angles(self.angles3, gripper_percentage=60)
            time.sleep(1)
            self.kinova_robot.set_joint_angles(self.angles4, gripper_percentage=60)
            time.sleep(1)
            self.kinova_robot.set_joint_angles(self.angles3, gripper_percentage=60)
            time.sleep(1)
            self.kinova_robot.set_joint_angles(self.angles2, gripper_percentage=0)
            time.sleep(1)
            # self.home = False

        else:
            self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)
            self.home = True            

if __name__ == "__main__":
    simulate = True
    
    if(simulate is None):
        raise ValueError("Pick simulate or real world robot")
    
    if simulate:
        final_project = Main(simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")
        # final_project = Main(simulate=True, urdf_path="visualizer/7dof/urdf/7dof.urdf")
        pass
    else:
        final_project = Main(is_suction=False)
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        final_project.shutdown()