from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robotx_2026',
            executable='led_node',
            output='screen',
        ),
        Node(
            package='robotx_2026',
            executable='pixhawk_led_node',
            output='screen',
        ),
    ])
