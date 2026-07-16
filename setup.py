from setuptools import setup, find_packages
from glob import glob

package_name = 'robotx_2026'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris',
    maintainer_email='chm018@ucsd.edu',
    description='Team Inspiration RobotX 2026 USV',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'led_node = robotx_2026.api.led.led_node:main',
            'pixhawk_led_node = robotx_2026.api.pixhawk.pixhawk_led_status_node:main',
            'gate_navigator = robotx_2026.api.navigation.gate_navigator:main',
            'dp_hold = robotx_2026.api.navigation.dp_hold:main',
        ],
    },
)
