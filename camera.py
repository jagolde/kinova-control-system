import cv2 as cv
from scipy import linalg
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
import copy
import matplotlib
matplotlib.use('TkAgg')

CUP_Z = 0
POS_SCALE = 1

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

        # 2D (x,y) correction computed during calibration refinement: world_xy = R @ raw_xy + t
        self.xy_correction = (np.eye(2), np.zeros(2))

    def find_all_markers(self, rgb, showIDs=False):
        """
        Detects ArUco/AprilTag markers and computes their poses
        using per-marker physical sizes.
        """

        gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

        aruco_dict = cv.aruco.getPredefinedDictionary(
            cv.aruco.DICT_APRILTAG_36H11
        )
        parameters = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

        # --- Detect once ---
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return

        ids = ids.flatten()

        if showIDs:
            cv.aruco.drawDetectedMarkers(rgb, corners, ids.reshape(-1, 1))
            plt.imshow(rgb)
            plt.show()

        # --- Define marker sizes ---
        marker_sizes = {
            0: 0.052,
            1: 0.052,
            2: 0.052,
            3: 0.052,
            5: 0.052,
            4: 0.1016,
            6: 0.1016,
            7: 0.1016
        }

        # --- Process each detected marker ---
        for i, marker_id in enumerate(ids):

            if marker_id not in marker_sizes:
                continue  # ignore unknown markers

            marker_size = marker_sizes[marker_id]

            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                [corners[i]], marker_size, self.K, self.distortion
            )

            rvec = rvecs[0]
            tvec = tvecs[0]

            R, _ = cv.Rodrigues(rvec)

            T = np.vstack(
                (np.hstack((R, tvec.reshape(3, 1))),
                [0, 0, 0, 1])
            )

            # store rotation + transform
            self.Rs[marker_id] = R
            self.Ts[marker_id] = T

            # camera position (already in camera frame)
            p_cam = tvec.reshape(3)

            world_pos = (self.cam_to_world @ np.append(p_cam, 1.0))[:3]

            # apply domain correction
            world_pos[2] += CUP_Z

            R_corr, t_corr = self.xy_correction
            world_pos[:2] = R_corr @ world_pos[:2] + t_corr

            self.world_positions[marker_id] = world_pos

    def calibrate_from_marker(self, marker_positions: dict, marker_size=0.1016):
        """
        Compute cam_to_world transform using known marker world positions
        and detected marker positions in camera frame.
        """

        # Get a fresh frame
        rgb, _ = self.get_frames()
        if rgb is None:
            raise RuntimeError("Failed to get frame for calibration")

        # Detect markers
        gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
        aruco_dict = cv.aruco.getPredefinedDictionary(
            cv.aruco.DICT_APRILTAG_36H11)
        detector = cv.aruco.ArucoDetector(aruco_dict)

        corners, ids, _ = detector.detectMarkers(gray)
        print("Detected IDs:", ids.flatten() if ids is not None else None)

        if ids is None:
            raise RuntimeError("No markers detected during calibration")

        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
            corners, marker_size, self.K, self.distortion
        )

        cam_pts = []
        world_pts = []

        for i, marker_id in enumerate(ids):
            mid = int(marker_id[0])
            if mid in marker_positions:
                cam_pts.append(tvecs[i][0])
                world_pts.append(marker_positions[mid])

        if len(cam_pts) < 3:
            raise RuntimeError("Need at least 3 markers for calibration")

        cam_pts = np.array(cam_pts)
        world_pts = np.array(world_pts)

        # --- Solve rigid transform (Kabsch algorithm) ---
        cam_centroid = np.mean(cam_pts, axis=0)
        world_centroid = np.mean(world_pts, axis=0)

        cam_centered = cam_pts - cam_centroid
        world_centered = world_pts - world_centroid

        H = cam_centered.T @ world_centered
        U, S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        # Fix reflection case
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = world_centroid - R @ cam_centroid

        # Build homogeneous transform
        self.cam_to_world = np.eye(4)
        self.cam_to_world[:3, :3] = R
        self.cam_to_world[:3, 3] = t

        # --- Optional XY correction (helps flatten small errors) ---
        raw_xy = []
        true_xy = []

        for i in range(len(cam_pts)):
            p_cam = np.append(cam_pts[i], 1.0)
            p_world_est = (self.cam_to_world @ p_cam)[:3]

            raw_xy.append(p_world_est[:2])
            true_xy.append(world_pts[i][:2])

        raw_xy = np.array(raw_xy)
        true_xy = np.array(true_xy)

        raw_centroid = np.mean(raw_xy, axis=0)
        true_centroid = np.mean(true_xy, axis=0)

        raw_centered = raw_xy - raw_centroid
        true_centered = true_xy - true_centroid

        H2 = raw_centered.T @ true_centered
        U2, S2, Vt2 = np.linalg.svd(H2)
        R2 = Vt2.T @ U2.T

        if np.linalg.det(R2) < 0:
            Vt2[1, :] *= -1
            R2 = Vt2.T @ U2.T

        t2 = true_centroid - R2 @ raw_centroid

        self.xy_correction = (R2, t2)

        print("Calibration complete")
        print("cam_to_world:\n", self.cam_to_world)

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
        intr = profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
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

        images = []

        for i in range(15):
            images.append(
                cv.imread(f"calibration_photos/img{i}.png", cv.IMREAD_COLOR))

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
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, gridshape, None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # Draw the corners
                image = copy.copy(image)
                corner_images.append(
                    cv.drawChessboardCorners(image, gridshape, corners2, ret)
                )
                valid.append(i)

        # calibrate the camera using the object points and image points
        ret, self.K, self.distortion, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )


class SimCamera(CameraBase):
    def __init__(self, kinova=None, fov_h=69.4, fov_v=42.5, w=640, h=480, nearLimit=0.01, farLimit=5, cameraPosition=[0.5, 0, 0.5], targetPosition=[0, 0, 0]):
        super().__init__(cameraPosition, w, h)

        self.targetPosition = targetPosition
        self.fov_h = fov_h  # Horizontal FOV in degrees
        self.fov_v = fov_v  # Vertical FOV in degrees
        self.nearLimit = nearLimit
        self.farLimit = farLimit

        self.kinova = kinova
        self.p = self.kinova.base_kinova.p

    def start(self):
        # Rotate camera position 180 degrees around target around Z-axis
        pos_rel = np.array(self.position) - np.array(self.targetPosition)
        # 180-degree rotation around Z-axis: (x, y, z) -> (-x, -y, z)
        pos_rel_rotated = np.array([-pos_rel[0], -pos_rel[1], pos_rel[2]])
        rotated_position = np.array(self.targetPosition) + pos_rel_rotated

        self.view = self.p.computeViewMatrix(
            rotated_position, self.targetPosition, [0, 0, 1])

        # Build frustum from both FOVs so the projection and K are consistent
        near = self.nearLimit
        right = near * np.tan(np.radians(self.fov_h / 2))
        top = near * np.tan(np.radians(self.fov_v / 2))
        self.proj = self.p.computeProjectionMatrix(
            -right, right, -top, top, near, self.farLimit)

        # K is derived from the same FOVs as the projection matrix
        fx = (self.w / 2) / np.tan(np.radians(self.fov_h / 2))
        fy = (self.h / 2) / np.tan(np.radians(self.fov_v / 2))

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
