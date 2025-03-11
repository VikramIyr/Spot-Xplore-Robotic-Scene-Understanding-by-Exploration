from __future__ import annotations

import time

import numpy as np

from bosdyn.client import Sdk
from robot_utils.advanced_movement import positional_grab, positional_release
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, move_body, set_gripper, stow_arm
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images
from utils.coordinates import Pose2D, Pose3D
from utils.logger import LoggerSingleton
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()
logger = LoggerSingleton()

STIFFNESS_DIAG1 = [200, 500, 500, 60, 60, 45]
STIFFNESS_DIAG2 = [100, 0, 0, 60, 30, 30]
DAMPING_DIAG = [2.5, 2.5, 2.5, 1.0, 1.0, 1.0]
FORCES = [0, 0, 0, 0, 0, 0]

class _BetterGrasp(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:
        ITEM = "green watering can"
        # ITEM = "poro plushy"

        if ITEM == "green watering can":
            grasp_coords = (0.222020, -2.652887, -0.037905)
            grasp_rot = np.asarray(
                [
                    [-0.253672, 0.672679, 0.695092],
                     [0.060881, 0.728276, -0.682575],
                     [-0.965372, -0.130832, -0.225697]

                ]
            )
            body_pose_distanced = Pose3D((0.777616, -2.111068, 0.365888))
        elif ITEM == "poro plushy":
            grasp_coords = (2.4, -2.5, 0.47)
            grasp_rot = np.asarray(
                [
                    [0.03121967, 0.60940186, -0.79224664],
                    [-0.99666667, -0.0407908, -0.07065172],
                    [-0.07537166, 0.79181155, 0.60609703],
                ]
            )
            body_pose_distanced = Pose3D((2.31, -1.33, 0.27))
        else:
            raise ValueError("Wrong ITEM!")

        RADIUS = 0.75
        RESOLUTION = 16
        logger.log(f"{ITEM=}", f"{RADIUS=}", f"{RESOLUTION=}")

        frame_name = localize_from_images(config, vis_block=False)
        start_pose_bosdyn = frame_transformer.get_current_body_position_in_frame(
            frame_name
        )
        start_pose = Pose2D.from_bosdyn_pose(start_pose_bosdyn)
        print(f"{start_pose=}")

        time.sleep(3)

        ###############################################################################
        ################################## PLANNING ###################################
        ###############################################################################

        grasp_pose_new = Pose3D(grasp_coords, grasp_rot)

        target_to_robot = np.asarray(grasp_coords) - body_pose_distanced.coordinates
        body_pose_distanced.set_rot_from_direction(target_to_robot)

        move_body(body_pose_distanced.to_dimension(2), frame_name)

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        # path = config.get_subpath("tmp")
        # pcd_path = os.path.join(path, "pcd.ply")
        # gripper_path = os.path.join(path, "gripper.ply")
        # pcd = o3d.io.read_point_cloud(pcd_path)
        # mesh = o3d.io.read_triangle_mesh(gripper_path)
        # o3d.visualization.draw_geometries([pcd, mesh])

        carry_arm(True)
        positional_grab(
            grasp_pose_new,
            0.25,
            -0.02,
            frame_name,
            already_gripping=False,
        )
        carry_arm(False)

        body_after = Pose3D((2, -1, 0))
        body_after2 = body_after.copy()
        body_after.rot_matrix = body_pose_distanced.rot_matrix.copy()
        move_body(body_after.to_dimension(2), frame_name)
        move_body(body_after2.to_dimension(2), frame_name)

        time.sleep(2)
        set_gripper(True)
        time.sleep(2)

        # # release
        # #get z coord of grab
        # z_coord = grasp_pose_new.coordinates[-1]
        # pose_hand = frame_transformer.get_hand_position_in_frame(frame_name, in_common_pose=True)
        # pose_hand.coordinates[-1] = z_coord
        #
        #
        # positional_release(pose=pose_hand,
        #                    frame_name=frame_name,
        #                    stiffness_diag_in=STIFFNESS_DIAG1,
        #                    stiffness_diag_out=STIFFNESS_DIAG2,
        #                    damping_diag_in=DAMPING_DIAG,
        #                    damping_diag_out=DAMPING_DIAG,
        #                    forces=FORCES,
        #                    follow_arm=True,
        #                    release_after=True)

        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_BetterGrasp(), body_assist=True)


if __name__ == "__main__":
    main()
