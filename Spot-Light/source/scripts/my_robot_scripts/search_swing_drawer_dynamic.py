# pylint: disable-all
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from bosdyn.api.image_pb2 import ImageResponse
from bosdyn.client import Sdk, math_helpers
from bosdyn.api import trajectory_pb2
from bosdyn.util import seconds_to_duration
from robot_utils.advanced_movement import pull, push, pull_swing_trajectory
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry, gaze, move_arm, move_body, stow_arm, set_gripper
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    GRIPPER_IMAGE_COLOR,
    frame_coordinate_from_depth_image,
    get_camera_rgbd,
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

from utils.pose_utils import (
    determine_handle_center,
    find_plane_normal_pose,
    calculate_handle_poses,
    cluster_handle_poses,
    filter_handle_poses,
    refine_handle_position,
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

def build_swing_trajectory(start_pose: Pose3D, lever: float, frame_name: str, positive_rotation: bool, roll:float=0, angle: int=80, N: int =5):
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

    rotX = euler_angles[2]
    rotY = euler_angles[1]
    rotZ = euler_angles[0]

    trajectory = []
    if positive_rotation == True:
        for angle in angles[1:]:
            delta_p_K = lever * np.array([-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)) - 1, 0])
            delta_p_W = start_pose.rot_matrix @ delta_p_K
            p_W = start_pose.coordinates + delta_p_W

            hand_pose = Pose3D(coordinates=p_W)
            hand_pose.set_rot_from_rpy((rotX, rotY, rotZ + angle), degrees=True)
            hand_pose.set_rot_from_direction(hand_pose.direction(), roll=roll, degrees=True)

            trajectory.append(hand_pose)

    if positive_rotation == False:
        for angle in angles[1:]:
            delta_p_K = lever * np.array([-np.sin(np.deg2rad(angle)), 1 - np.cos(np.deg2rad(angle)), 0])
            delta_p_W = start_pose.rot_matrix @ delta_p_K
            p_W = start_pose.coordinates + delta_p_W

            hand_pose = Pose3D(coordinates=p_W)
            hand_pose.set_rot_from_rpy((rotX, rotY, rotZ - angle), degrees=True)
            hand_pose.set_rot_from_direction(hand_pose.direction(), roll=roll, degrees=True)

            trajectory.append(hand_pose)

    return trajectory

def determine_handle_center(
    depth_image: np.ndarray, bbox: BBox, approach: str = "center"
) -> np.ndarray:
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    if approach == "min":
        image_patch = depth_image[xmin:xmax, ymin:ymax].squeeze()
        image_patch[image_patch == 0.0] = 100_000
        center_flat = np.argmin(depth_image[xmin:xmax, ymin:ymax])
        center = np.array(np.unravel_index(center_flat, image_patch.shape))
        center = center.reshape((2,)) + np.array([xmin, ymin]).reshape((2,))
    elif approach == "center":
        x_center, y_center = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
        center = np.array([x_center, y_center])
    else:
        raise ValueError(f"Unknown type {approach}. Must be either 'min' or 'center'.")
    return center


def find_plane_normal_pose(
    points: np.ndarray,
    center_coords: np.ndarray,
    current_body: Pose2D,
    threshold: float = 0.04,
    min_samples: int = 3,
    vis_block: bool = False,
) -> Pose3D:

    normal = plane_fitting_open3d(
        points, threshold=threshold, min_samples=min_samples, vis_block=vis_block
    )
    # dot product between offset and normal is negative when they point in opposite directions
    offset = center_coords[:2] - current_body.coordinates
    sign = np.sign(np.dot(offset, normal[:2]))
    normal = sign * normal
    pose = Pose3D(center_coords)
    pose.set_rot_from_direction(normal)
    return pose

def calculate_swing_params(
    matches: list[Match],
    depth_image_response: (np.ndarray, ImageResponse),
    frame_name: str,
    ) -> tuple(list[bool]):

    """
    Calculates rotation directionm and swing lever of all swing doors in the image.
    """

    positive_rotation = []
    metric_lever = []

    drawer_boxes = [match.drawer.bbox for match in matches]
    handle_boxes = [match.handle.bbox for match in matches]

    depth_image, depth_response = depth_image_response

    # determine center coordinates for all handles and drawers
    for idx, _ in enumerate(handle_boxes):
        handle_center = determine_handle_center(depth_image, handle_boxes[idx])
        drawer_center = determine_handle_center(depth_image, drawer_boxes[idx])

        # calcualte drawer xmin and xmax coords for lever calculation
        pixel_coord_lever = np.array([[int(drawer_boxes[idx].xmin), int((drawer_boxes[idx].ymin + drawer_boxes[idx].ymax) / 2)],
                                           [int(drawer_boxes[idx].xmax), int((drawer_boxes[idx].ymin + drawer_boxes[idx].ymax) / 2)],
                                           handle_center])

        frame_coord_drawer_min = frame_coordinate_from_depth_image(depth_image=depth_image,
                                                                depth_response=depth_response,
                                                                pixel_coordinatess=pixel_coord_lever,
                                                                frame_name=frame_name,
                                                                vis_block=False,
                                                                ).reshape((-1, 3))

        if handle_center[0] > drawer_center[0]:
            positive_rotation.append(False)
            metric_lever.append(np.linalg.norm(frame_coord_drawer_min[0,:]-frame_coord_drawer_min[2,:]))
        elif handle_center[0] < drawer_center[0]:
            positive_rotation.append(True)
            metric_lever.append(np.linalg.norm(frame_coord_drawer_min[1, :] - frame_coord_drawer_min[2, :]))

    return positive_rotation, metric_lever


def calculate_handle_poses(
    matches: list[Match],
    depth_image_response: (np.ndarray, ImageResponse),
    frame_name: str,
) -> list[Pose3D]:
    """
    Calculates pose and axis of motion of all handles in the image.
    """
    centers = []
    drawer_boxes = [match.drawer.bbox for match in matches]
    handle_boxes = [match.handle.bbox for match in matches]

    depth_image, depth_response = depth_image_response

    # determine center coordinates for all handles
    for handle_bbox in handle_boxes:
        center = determine_handle_center(depth_image, handle_bbox)
        centers.append(center)
    if len(centers) == 0:
        return []
    centers = np.stack(centers, axis=0)

    # use centers to get depth and position of handle in frame coordinates
    center_coordss = frame_coordinate_from_depth_image(
        depth_image=depth_image,
        depth_response=depth_response,
        pixel_coordinatess=centers,
        frame_name=frame_name,
        vis_block=False,
    ).reshape((-1, 3))

    # select all points within the point cloud that belong to a drawer (not a handle) and determine the planes
    # the axis of motion is simply the normal of that plane
    drawer_bbox_pointss = select_points_from_bounding_box(
        depth_image_response, drawer_boxes, frame_name, vis_block=False
    )
    handle_bbox_pointss = select_points_from_bounding_box(
        depth_image_response, handle_boxes, frame_name, vis_block=False
    )
    points_frame = drawer_bbox_pointss[0]
    drawer_masks = drawer_bbox_pointss[1]
    handle_masks = handle_bbox_pointss[1]
    drawer_only_masks = drawer_masks & (~handle_masks)
    # for mask in drawer_only_masks:
    #     vis.show_point_cloud_in_out(points_frame, mask)

    # we use the current body position to get the normal that points towards the robot, not away
    current_body = frame_transformer.get_current_body_position_in_frame(
        frame_name, in_common_pose=True
    )
    poses = []
    for center_coords, bbox_mask in zip(center_coordss, drawer_only_masks):
        pose = find_plane_normal_pose(
            points_frame[bbox_mask],
            center_coords,
            current_body,
            threshold=0.03,
            min_samples=10,
            vis_block=False,
        )
        poses.append(pose)

    return poses

def cluster_handle_poses(
    handles_posess: list[list[Pose3D]],
    eps: float = MIN_PAIRWISE_DRAWER_DISTANCE,
    min_samples: int = 2,
) -> list[Pose3D]:
    handles_poses_flat = [
        handle_pose for handles_poses in handles_posess for handle_pose in handles_poses
    ]
    handle_coords = [handle_pose.coordinates for handle_pose in handles_poses_flat]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(handle_coords)

    cluster_dict = defaultdict(list)
    for idx, label in enumerate(dbscan.labels_):
        handle_pose = handles_poses_flat[idx]
        cluster_dict[str(label)].append(handle_pose)
    print("cluster_dict=", *cluster_dict.items(), sep="\n")

    avg_poses = []
    for key, cluster in cluster_dict.items():
        if key == "-1":
            continue
        avg_pose = average_pose3Ds(cluster)
        avg_poses.append(avg_pose)
    return avg_poses

def filter_handle_poses(handle_poses: list[Pose3D], positive_rotations: list[bool], metric_levers: list[float]):

    filtered_rotations = []
    filtered_poses = []
    filtered_levers = []
    for idx, pose in enumerate(handle_poses):
        if 0.05 < pose.coordinates[-1] < 0.75:
            filtered_poses.append(pose)
            filtered_rotations.append(positive_rotations[idx])
            filtered_levers.append(metric_levers[idx])

    return filtered_poses, filtered_rotations, filtered_levers

def refine_handle_position(
    handle_detections: list[Detection],
    prev_pose: Pose3D,
    depth_image_response: (np.ndarray, ImageResponse),
    frame_name: str,
    discard_threshold: int = MIN_PAIRWISE_DRAWER_DISTANCE,
) -> (Pose3D, bool):
    depth_image, depth_response = depth_image_response
    prev_center_3D = prev_pose.coordinates.reshape((1, 3))

    if len(handle_detections) == 0:
        warnings.warn("No handles detected in refinement!")
        return prev_pose, True
    elif len(handle_detections) > 1:
        centers_2D = []
        for det in handle_detections:
            handle_bbox = det.bbox
            center = determine_handle_center(depth_image, handle_bbox)
            centers_2D.append(center)
        centers_2D = np.stack(centers_2D, axis=0)
        centers_3D = frame_coordinate_from_depth_image(
            depth_image, depth_response, centers_2D, frame_name, vis_block=False
        )
        closest_new_idx = np.nanargmin(
            np.linalg.norm(centers_3D - prev_center_3D, axis=1), axis=0
        )
        handle_bbox = handle_detections[closest_new_idx].bbox
        detection_coordinates_3D = centers_3D[closest_new_idx].reshape((1, 3))
    else:
        handle_bbox = handle_detections[0].bbox
        center = determine_handle_center(depth_image, handle_bbox).reshape((1, 2))
        detection_coordinates_3D = frame_coordinate_from_depth_image(
            depth_image, depth_response, center, frame_name, vis_block=False
        )

    # if the distance between expected and mean detection is too large, it likely means that we detect another
    # and also do not detect the original one we want to detect -> discard
    handle_offset = np.linalg.norm(detection_coordinates_3D - prev_center_3D)
    discarded = False
    if handle_offset > discard_threshold:
        discarded = True
        print(
            "Only detection discarded as unlikely to be true detection!",
            f"{handle_offset=}",
            sep="\n",
        )
        detection_coordinates_3D = prev_center_3D

    xmin, ymin, xmax, ymax = handle_bbox
    d = 40
    surrounding_bbox = BBox(xmin - d, ymin - d, xmax + d, ymax + d)
    points_frame, [handle_mask, surr_mask] = select_points_from_bounding_box(
        depth_image_response,
        [handle_bbox, surrounding_bbox],
        frame_name,
        vis_block=True,
    )
    surr_only_mask = surr_mask & (~handle_mask)
    current_body = frame_transformer.get_current_body_position_in_frame(
        frame_name, in_common_pose=True
    )

    detection_coordinates_3D = detection_coordinates_3D.reshape((3,))
    pose = find_plane_normal_pose(
        points_frame[surr_only_mask],
        detection_coordinates_3D,
        current_body,
        threshold=0.04,
        min_samples=10,
        vis_block=False,
    )

    # PCA
    import open3d as o3d

    pcd = points_frame[handle_mask]
    points = pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mean, cov = pcd.compute_mean_and_covariance()
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    PA_w = eigenvectors[:,0]

    # Compute the angle between the principal axisand the Z axis of world system
    roll_angle = np.degrees(np.pi/2 - np.arccos(np.dot(PA_w, np.array([0, 0, 1])) / (
                np.linalg.norm(PA_w) * np.linalg.norm(np.array([0, 0, 1])))))

    return pose, discarded, roll_angle

def calc_swing_door_rotation(pose, discarded):
    pass

class _DynamicSwingDoor(ControlFunction):
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
        x, y, angle = 1.2, -1.0, 270 # large cabinet, +z
        # x, y, angle = 1.1, -1.2, 270  # large cabinet, +z

        pose = Pose2D(np.array([x, y]))
        pose.set_rot_from_angle(angle, degrees=True)
        move_body(
            pose=pose,
            frame_name=frame_name,
        )

        # set initial arm coords
        # ground truth coord of handle
        cabinet_pose = Pose3D((1.20, -2.5, 0.55))
        # cabinet_pose = Pose3D((1.35, -2.45, 0.61)) # large cabinet, +z
        cabinet_pose.set_rot_from_rpy((0,0,angle), degrees=True)
        # carry()
        # move_arm(knob_pose, frame_name, body_assist=True)

        # detect drawer and handle

        carry()
        gaze(cabinet_pose, frame_name, gripper_open=True)
        depth_response, color_response = get_camera_rgbd(
            in_frame="image",
            vis_block=False,
            cut_to_size=False,
        )
        stow_arm()
        predictions = drawer_predict(
            color_response[0], config, input_format="bgr", vis_block=True
        )

        handle_posess = []
        matches = drawer_handle_matches(predictions)
        filtered_matches = [
            m for m in matches if (m.handle is not None and m.drawer is not None)
        ]
        filtered_sorted_matches = sorted(
            filtered_matches, key=lambda m: (m.handle.bbox.ymin, m.handle.bbox.xmin)
        )
        handle_poses = calculate_handle_poses(
            filtered_sorted_matches, depth_response, frame_name
        )
        # handle_posess.append(handle_poses)
        positive_rotations, metric_levers = calculate_swing_params(filtered_sorted_matches, depth_response, frame_name)
        # print("all detections:", *handle_posess, sep="\n")
        # handle_poses = cluster_handle_poses(handle_posess, eps=MIN_PAIRWISE_DRAWER_DISTANCE)
        # print("clustered:", *handle_poses, sep="\n")
        handle_poses, positive_rotations, metric_levers = filter_handle_poses(handle_poses, positive_rotations, metric_levers)
        print("filtered:", *handle_poses, sep="\n")

        camera_add_pose_refinement_right = Pose3D((-0.35, -0.2, 0.15))
        camera_add_pose_refinement_right.set_rot_from_rpy((0, 25, 35), degrees=True)
        camera_add_pose_refinement_left = Pose3D((-0.35, 0.2, 0.15))
        camera_add_pose_refinement_left.set_rot_from_rpy((0, 25, -35), degrees=True)
        ref_add_poses = [camera_add_pose_refinement_right]#, camera_add_pose_refinement_left)

        detection_drawer_pairs = []

        carry()

        for idx, handle_pose in enumerate(handle_poses):
            # if no handle is detected in the refinement, redo it from a different position
            body_pose = pose_distanced(handle_pose, STAND_DISTANCE).to_dimension(2)
            move_body(body_pose, frame_name)

            refined_pose = handle_pose
            for ref_pose in ref_add_poses:
                move_arm(handle_pose @ ref_pose, frame_name)
                depth_response, color_response = get_camera_rgbd(
                    in_frame="image", vis_block=False, cut_to_size=False
                )
                predictions = drawer_predict(
                    color_response[0], config, input_format="bgr", vis_block=True
                )

                handle_detections = [det for det in predictions if det.name == "handle"]

                refined_pose, discarded, roll_angle = refine_handle_position(
                    handle_detections, handle_pose, depth_response, frame_name
                )

                a = 2

            print(f"{refined_pose=}")

            a = 2
            traj = build_swing_trajectory(start_pose=refined_pose,
                                              lever=metric_levers[idx],
                                              frame_name=frame_name,
                                              positive_rotation=positive_rotations[idx],
                                              roll=roll_angle,
                                              angle=90,
                                              N=2)

            refined_pose.set_rot_from_direction(refined_pose.direction(), roll=roll_angle, degrees=True)

            pull_swing_trajectory(pose=refined_pose,
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
    take_control_with_function(config, function=_DynamicSwingDoor(), return_to_start=True)


if __name__ == "__main__":
    main()
