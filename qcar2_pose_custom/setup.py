from setuptools import find_packages, setup

package_name = 'qcar2_pose_custom'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/slam_toolbox_config.yaml']),
        ('share/' + package_name + '/launch', ['launch/autonomy_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adithya',
    maintainer_email='adithya@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "pose_node = qcar2_pose_custom.node1:main",
            "temp_node = qcar2_pose_custom.node2:main",
            "odom_publisher = qcar2_pose_custom.node3:main",
            "static_tf_publisher = qcar2_pose_custom.node4:main",
            "global_pose_publisher = qcar2_pose_custom.node5:main",
            "keyboard_control = qcar2_pose_custom.node6:main",
            "odom_publisher_ekf = qcar2_pose_custom.node7:main",
            "odom_from_tf = qcar2_pose_custom.node8:main",
        ],
    },
)
