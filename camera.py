import pyrealsense2 as rs
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy import linalg
import cv2 as cv

class CameraBase:
    def __init__(self, cameraPosition, w, h):

        # Basic camera information
        self.position = cameraPosition
        self.w, self.h, = w, h
        self.aspect = self.w / self.h

        # Different depending on sim or realsense
        self.K = None
        self.distortion = None
        self.cam_to_world = None  # set by subclass in start()

        # List of all IDs and dictionaries of all transformation, rotation, and world position for each
        self.ids = []
        self.Ts = dict()
        self.Rs = dict()
        self.world_positions = dict()

    def find_all_markers(self, rgb):
        '''
        Finds aruco markers within the rgb image
        '''
        gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
        aruco_dict = cv.aruco.getPredefinedDictionary(
            cv.aruco.DICT_APRILTAG_36H11)
        parameters = cv.aruco.DetectorParameters()

        # Create the ArUco detector
        detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
        # Detect the markers
        corners, self.ids, _ = detector.detectMarkers(gray)

        if self.ids is not None:
            cv.aruco.drawDetectedMarkers(rgb, corners, self.ids)
            plt.imshow(rgb)
            plt.show()

        if self.ids is not None:
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                corners, 0.05, self.K, self.distortion
            )

            for i, marker_id in enumerate(self.ids):
                mid = int(marker_id[0])
                R, _ = cv.Rodrigues(rvecs[i])
                T = np.vstack(
                    (np.hstack((R, tvecs[i].reshape(3, 1))), [0, 0, 0, 1]))

                self.Rs[mid] = R
                self.Ts[mid] = T

                p_cam = np.append(tvecs[i][0], 1.0)
                self.world_positions[mid] = (self.cam_to_world @ p_cam)[:3]

    def undistort(self, img):
        return cv.undistort(img, self.K, self.distortion, None, None)

    def start(self):
        return None

    def stop(self):
        pass


class RealsenseCamera(CameraBase):
    @staticmethod
    def is_connected():
        return len(rs.context().devices) > 0

    def __init__(self, cameraPosition=[0.5, 0, 0.5], orientation=np.eye(3), w=640, h=480, fps=30):
        super().__init__(cameraPosition, w, h)
        self.fps = fps
        self.pipeline = None
        self.align = None

        # Build cam_to_world from position + orientation (rotation matrix, camera frame -> world frame)
        self.cam_to_world = np.eye(4)
        self.cam_to_world[:3, :3] = orientation
        self.cam_to_world[:3, 3] = cameraPosition

    def start(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.w,
                             self.h, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.w,
                             self.h, rs.format.z16, self.fps)
        profile = self.pipeline.start(config)

        # Align the depth frame to the color frame for easier pixel correspondence
        self.align = rs.align(rs.stream.color)

        # Read factory intrinsics directly from the camera
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], dtype=np.float64)
        self.distortion = np.array(intr.coeffs, dtype=np.float64)

        # Discard frames while auto-exposure and auto-white-balance settle
        for _ in range(30):
            self.pipeline.wait_for_frames()

        return profile

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            self.align = None

    def get_frames(self, timeout_ms=5000):
        if self.pipeline is None:
            raise RuntimeError("Realsense pipeline is not started")

        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned_frames = self.align.process(
            frames) if self.align is not None else frames

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def calibration(self):
        '''
        Calibrates the camera to adjust for distortion
        '''
        images = ImageCollection("./calibration_photos/*.png")

        gridshape = (9, 6)
        squaresize = 24e-3

        criteria = (cv.TERM_CRITERIA_EPS +
                    cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # create set of feature points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # these all have Z=0 since they are relative to the calibration target frame
        objp = np.zeros((gridshape[0] * gridshape[1], 3), np.float32)
        objp[:, :2] = (
            np.mgrid[0: gridshape[0], 0: gridshape[1]
                     ].T.reshape(-1, 2) * squaresize
        )

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        corner_images = []
        valid = []

        # loop through the images and find the corners
        for i, image in enumerate(images):
            gray = image.mono().A

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, gridshape, None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # Draw the corners
                image = Image(image, copy=True)
                if not image.iscolor:
                    image = image.colorize()
                corner_images.append(
                    cv.drawChessboardCorners(image.A, gridshape, corners2, ret)
                )
                valid.append(i)

        # calibrate the camera using the object points and image points
        ret, self.K, self.distortion, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )


class SimCamera(CameraBase):
    def __init__(self, kinova=None, fov=60, w=640, h=480, nearLimit=0.01, farLimit=5, cameraPosition=[0.5, 0, 0.5], targetPosition=[0, 0, 0]):
        super().__init__(cameraPosition, w, h)

        self.targetPosition = targetPosition
        self.fov = fov
        self.nearLimit = nearLimit
        self.farLimit = farLimit

        self.kinova = kinova
        self.p = self.kinova.base_kinova.p

    def start(self):
        self.view = self.p.computeViewMatrix(
            self.position, self.targetPosition, [0, 0, 1])
        self.proj = self.p.computeProjectionMatrixFOV(
            self.fov, self.w/self.h, self.nearLimit, self.farLimit)

        fy = (self.h / 2) / np.tan(np.radians(self.fov / 2))
        fx = (self.w / 2) / (self.aspect * np.tan(np.radians(self.fov / 2)))

        self.K = np.array([[fx, 0, self.w / 2], [0, fy, self.h / 2],
                          [0, 0, 1]], dtype=np.float64)
        self.distortion = np.zeros((4, 1))

        # PyBullet view matrix is 16-element column-major: world -> OpenGL camera
        # OpenGL: Y-up, Z-backward.  OpenCV: Y-down, Z-forward.  Flip Y and Z.
        V = np.array(self.view).reshape(4, 4).T
        M_flip = np.diag([1.0, -1.0, -1.0, 1.0])
        self.cam_to_world = np.linalg.inv(V) @ M_flip

        return None

    def get_frames(self):
        '''
        Gets frames from Simulation camera and removes distortion if calibration has occurred
        '''

        w, h, rgba, depth, seg = self.p.getCameraImage(
            640, 480, self.view, self.proj)
        rgb = np.array(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, 2::-1]
        return rgb, depth
    



# remove distortion if needed
# if self.k is None or self.distortion is None:
#     rgba = cv.undistort(rgba, self.K, self.distortion, None, None)
#     depth = cv.undistort(depth, self.K, self.distortion, None, None)
