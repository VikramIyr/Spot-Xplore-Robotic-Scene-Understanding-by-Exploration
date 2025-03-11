# pylint: disable-all
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client import Sdk, math_helpers
from bosdyn.api import trajectory_pb2
from bosdyn.util import seconds_to_duration
from robot_utils.advanced_movement import pull, push, pull_swing_trajectory, pull_swing_trajectory_test
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry, gaze, move_arm, move_body, stow_arm, set_gripper
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    GRIPPER_IMAGE_COLOR,
    frame_coordinate_from_depth_image,
    get_camera_rgbd,
    get_d_pictures,
    get_rgb_pictures,
    localize_from_images,
    select_points_from_bounding_box,
)
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN, KMeans
from utils import recursive_config
from utils.camera_geometry import plane_fitting_open3d
from utils.coordinates import Pose2D, Pose3D, average_pose3Ds, pose_distanced
from utils.drawer_detection import drawer_handle_matches
from utils.drawer_detection import predict_yolodrawer as drawer_predict
from utils.importer import PointCloud
from utils.object_detetion import BBox, Detection, Match
from utils.openmask_interface import get_mask_points
from utils.point_clouds import body_planning_mult_furthest
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)
from utils.zero_shot_object_detection import detect_objects

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()

STAND_DISTANCE = 1.0
STIFFNESS_DIAG1 = [200, 500, 500, 60, 60, 45]
STIFFNESS_DIAG2 = [100, 0, 0, 60, 30, 30]
DAMPING_DIAG = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
FORCES = [0, 0, 0, 0, 0, 0]
CAMERA_ADD_COORDS = (-0.25, 0, 0.3)
CAMERA_ANGLE = 55
SPLIT_THRESH = 1.0
MIN_PAIRWISE_DRAWER_DISTANCE = 0.1
ITEMS = ["deer toy", "small clock", "headphones", "watch", "highlighter", "red bottle"]
# ITEMS = ["watch"]

def build_swing_trajectory(start_pose: Pose3D, lever: float, frame_name: str, positive_rotation: bool, angle: int=80, N: int =5):
    """
    build a swing trajectory of Pose3D poses following the handle of a swing door

    @param start_pose: pose of the knob in world COS. start of the trajectory.
    @param lever: distance from knob to hinges (pivot point) in meter
    @param frame_name:
    @param positive_rotation: indicates the direction of rotation around the hinges. With Z axis pointing upwards, positive rotaion is given by right hand rule.
    @param angle: desired opening angle of the door
    @param N: number of trajectory points. NOTE: using impedance control, one might be sufficient.
    @return: trajectory, List of Pose3D objects
    """

    # Define angles for each point in the trajectory.
    angles = np.linspace(0,angle,N+1)

    # return z rotation of the knob pose
    r = Rotation.from_matrix(start_pose.rot_matrix)
    euler_angles = r.as_euler("zyx", degrees=True)
    rot = euler_angles[0]

    trajectory = []
    if positive_rotation == True:
        for angle in angles[1:]:
            delta_p_K = lever * np.array([-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)) - 1, 0])
            delta_p_W = start_pose.rot_matrix @ delta_p_K
            p_W = start_pose.coordinates + delta_p_W

            hand_pose = Pose3D(coordinates=p_W)
            hand_pose.set_rot_from_rpy((0, 0, rot + angle), degrees=True)

            trajectory.append(hand_pose)

    if positive_rotation == False:
        for angle in angles[1:]:
            delta_p_K = lever * np.array([-np.sin(np.deg2rad(angle)), 1 - np.cos(np.deg2rad(angle)), 0])
            delta_p_W = start_pose.rot_matrix @ delta_p_K
            p_W = start_pose.coordinates + delta_p_W

            hand_pose = Pose3D(coordinates=p_W)
            hand_pose.set_rot_from_rpy((0, 0, rot - angle), degrees=True)

            trajectory.append(hand_pose)

    return trajectory


class _StaticSwingDoor(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        indices = (1,)
        config = recursive_config.Config()

        frame_name = localize_from_images(config, vis_block=False)
        start_pose = frame_transformer.get_current_body_position_in_frame(
            frame_name, in_common_pose=True
        )

        # set rotation (depends on the swing door -> left swing (negative) / right swing (positive))
        positive_rotation = False

        # position in front of shelf
        # x, y, angle = 1.35, 0.7, 180 #high cabinet, -z
        x, y, angle = 1.65, -1.4, 270#-1.5 #large cabinet
        # x, y, angle = 1.25, -1.5, 270 # large cabinet, +z

        pose = Pose2D(np.array([x, y]))
        pose.set_rot_from_angle(angle, degrees=True)
        move_body(
            pose=pose,
            frame_name=frame_name,
        )

        # set initial arm coords
        # knob_pose = Pose3D((0.35, 0.8, 0.75)) #high cabinet
        knob_pose = Pose3D((1.50, -2.45, 0.57))#-2.47 #large cabinet
        # knob_pose = Pose3D((1.45, -2.45, 0.57)) # large cabinet, +z
        knob_pose.set_rot_from_rpy((0,0,angle), degrees=True)
        # arm_pose.set_rot_from_rpy((0, 0, 180), degrees=True)
        # set_gripper(True)
        # # set arm pos
        # carry()
        # move_arm(knob_pose, frame_name, body_assist=True)

        a = 2



        # calc trajectory
        traj = build_swing_trajectory(start_pose=knob_pose,
                                     lever=0.33,
                                     frame_name=frame_name,
                                     positive_rotation=positive_rotation,
                                     angle=90,
                                     N=1)

        pull_swing_trajectory_test(pose=knob_pose,
                              start_distance=0.1,
                              mid_distance=-0.05,
                              trajectory=traj,
                              frame_name=frame_name,
                              stiffness_diag_in=STIFFNESS_DIAG1,
                              damping_diag_in=DAMPING_DIAG,
                              stiffness_diag_out=STIFFNESS_DIAG2,
                              damping_diag_out=DAMPING_DIAG,
                              forces=FORCES,
                              follow_arm=False,
                              release_after=True
                              )

        stow_arm()


        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_StaticSwingDoor(), return_to_start=True)


if __name__ == "__main__":
    main()
