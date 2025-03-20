import argparse
import sys
import cv2
import numpy as np
import os
import json
import time
from bosdyn.api import image_pb2, gripper_camera_param_pb2
from bosdyn.client import create_standard_sdk, util
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)

DATASET_DIR = 'dataset/realsense'
COLOR_DIR = os.path.join(DATASET_DIR, 'color')
DEPTH_DIR = os.path.join(DATASET_DIR, 'depth')
INTRINSICS_FILE = os.path.join(DATASET_DIR, 'camera_intrinsic.json')

os.makedirs(COLOR_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

IMAGE_SOURCES = {
    'hand_color_image',
    'hand_depth_in_hand_color_frame'
}

def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]

def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)

def get_next_index(directory, extension):
    existing_files = [f for f in os.listdir(directory) if f.endswith(extension)]
    indices = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
    return max(indices, default=-1) + 1

def save_camera_intrinsics(width, height, fx, fy, cx, cy):
    intrinsics_data = {
        "width": width,
        "height": height,
        "intrinsic_matrix": [fx, 0, 0, 0, fy, 0, cx, cy, 1]
    }
    with open(INTRINSICS_FILE, 'w') as f:
        json.dump(intrinsics_data, f, indent=4)

def capture_and_save_images(image_client, pixel_format):
    image_request = [
        build_image_request(image_source_name=source, quality_percent=100, pixel_format=pixel_format)
        for source in IMAGE_SOURCES
    ]
    image_responses = image_client.get_image(image_request)

    for image in image_responses:
        width = image.shot.image.cols
        height = image.shot.image.rows
        fx = image.source.pinhole.intrinsics.focal_length.x
        fy = image.source.pinhole.intrinsics.focal_length.y
        cx = image.source.pinhole.intrinsics.principal_point.x
        cy = image.source.pinhole.intrinsics.principal_point.y

        save_camera_intrinsics(width, height, fx, fy, cx, cy)

        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
            extension = '.png'
            save_dir = DEPTH_DIR
        else:
            dtype = np.uint8
            extension = '.jpg'
            save_dir = COLOR_DIR

        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                img = img.reshape((height, width, -1))
            except ValueError:
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        index = get_next_index(save_dir, extension)
        filename = f"{index:06d}{extension}"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"Image saved as {filepath}.")

def main():
    parser = argparse.ArgumentParser()
    util.add_base_arguments(parser)
    parser.add_argument('--pixel-format', choices=pixel_format_type_strings(),
                        help='Requested pixel format of image. If supplied, used for all sources.')
    parser.add_argument('--capture-interval', type=float, default=0.1,
                        help='Interval between captures in video mode, in seconds.')
    options = parser.parse_args()

    sdk = create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)

    camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_640_480
    params = gripper_camera_param_pb2.GripperCameraParams(camera_mode=camera_mode)

    gripper_camera_param_client = robot.ensure_client(GripperCameraParamClient.default_service_name)
    gripper_camera_param_client.set_camera_params(gripper_camera_param_pb2.GripperCameraParamRequest(params=params))

    pixel_format = pixel_format_string_to_enum(options.pixel_format)

    # Make the open gripper RobotCommand
    # command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    # gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
    # command = RobotCommandBuilder.build_synchro_command(gripper_command)
    # command_client.robot_command(command)

    print("Press (c) for single capture mode")
    print("Press (v) for video capture mode")
    print("Press (q) to quit the program")

    while True:
        user_input = input("Enter command (c/v/q): ").strip().lower()
        if user_input == 'c':
            capture_and_save_images(image_client, pixel_format)
        elif user_input == 'v':
            print("Entering video capture mode. Press Ctrl+C or 'q' to stop.")
            try:
                while True:
                    capture_and_save_images(image_client, pixel_format)
                    time.sleep(options.capture_interval)
            except KeyboardInterrupt:
                print("Video capture mode stopped by user.")
            except Exception as e:
                print(f"Error during video capture mode: {e}")
        elif user_input == 'q':
            # gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)
            # command = RobotCommandBuilder.build_synchro_command(gripper_command)
            # command_client.robot_command(command)
            print("Quitting program.")
            break
        else:
            print("Invalid input, please press 'c', 'v', or 'q'.")

if __name__ == '__main__':
    if not main():
        sys.exit(1)
