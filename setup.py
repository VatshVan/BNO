import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'bno'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vatshvan',
    maintainer_email='vatshvan.iitb@gmail.com',
    description='Extended Kalman Filter for BNO055 IMU and GNSS fusion',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bno_node = bno.bno_node:main',
            'gps_node = bno.gps_node:main',
            'ekf_ros_node = bno.ekf_ros_node:main',
        ],
    },
)
