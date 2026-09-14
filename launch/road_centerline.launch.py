"""Launch usb_cam and road_centerline_node together.

usb_cam publishes on the relative topic "image_raw", which resolves to
"/image_raw" since neither node here is namespaced — this must match the
"image_topic" parameter in road_centerline_params.yaml.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory('road_centerline'),
        'config', 'road_centerline_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params_file,
        description='Path to the road_centerline_node parameters yaml file')

    video_device_arg = DeclareLaunchArgument(
        'video_device', default_value='/dev/video33',
        description='V4L2 device path for the USB camera. Run `v4l2-ctl '
                     '--list-devices` to find yours — /dev/video0 and low-numbered '
                     'devices are often virtual (e.g. v4l2loopback) or ISP nodes, '
                     'not the real UVC camera.')
    image_width_arg = DeclareLaunchArgument(
        'image_width', default_value='160',
        description='Camera capture width in pixels — matches the model input '
                     'size, so road_centerline_node does not need to resize. '
                     'Not all cameras support this resolution natively; check '
                     '`v4l2-ctl --list-formats-ext` and override if needed.')
    image_height_arg = DeclareLaunchArgument(
        'image_height', default_value='120',
        description='Camera capture height in pixels — matches the model input size')
    framerate_arg = DeclareLaunchArgument(
        'framerate', default_value='30.0',
        description='Camera capture framerate')
    pixel_format_arg = DeclareLaunchArgument(
        'pixel_format', default_value='yuyv',
        description='usb_cam pixel format — check `v4l2-ctl --list-formats-ext` '
                     'for what your camera actually supports')

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('video_device'),
            'image_width': ParameterValue(LaunchConfiguration('image_width'), value_type=int),
            'image_height': ParameterValue(LaunchConfiguration('image_height'), value_type=int),
            'framerate': ParameterValue(LaunchConfiguration('framerate'), value_type=float),
            'pixel_format': LaunchConfiguration('pixel_format'),
            'camera_name': 'usb_cam',
            'frame_id': 'camera',
        }],
    )

    road_centerline_node = Node(
        package='road_centerline',
        executable='road_centerline_node',
        name='road_centerline_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([
        params_file_arg,
        video_device_arg,
        image_width_arg,
        image_height_arg,
        framerate_arg,
        pixel_format_arg,
        usb_cam_node,
        road_centerline_node,
    ])
