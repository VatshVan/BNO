import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'bno'
    
    config_file_path = os.path.join(
        get_package_share_directory(pkg_name),
        'params',
        'ekf_params.yaml'
    )

    ekf_node = Node(
        package=pkg_name,
        executable='ekf_ros_node',
        name='ekf_fusion_node',
        parameters=[config_file_path],
        output='screen'
    )

    imu_node = Node(
        package=pkg_name,
        executable='bno_node',
        name='imu_serial_driver',
        parameters=[config_file_path],
        output='screen'
    )

    gps_node = Node(
        package=pkg_name,
        executable='gps_node',
        name='gnss_serial_driver',
        parameters=[config_file_path],
        output='screen'
    )

    return LaunchDescription([
        ekf_node,
        imu_node,
        gps_node
    ])