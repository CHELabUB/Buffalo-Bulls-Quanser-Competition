# Buffalo-Bulls-Quanser-Competition
This repo is build to document the codes used in ACC competition in 2026. 

# Activate the testing environment (assume that the installation of the containers are done properly) 
Following the instructions on the webpage: [Development Guide for the Development Container ](https://github.com/quanser/student-competition-resources-ros/blob/main/Virtual_ROS_Resources/Virtual_ROS_Development_Guide.md)
## How to run
### 1 Start Qlabs
```bash
/usr/bin/QLabs
```
### 2 Open a new terminal and launch the competition simulaiton
```bash
docker exec -it virtual-qcar2 bash
cd /home/qcar2_scripts/python
python3 Base_Scenarios_Python/Setup_Competition_Map.py
```
### 3 Open a new terminal and launch Qcar2
```bash
cd /home/$USER/Documents/ACC_Development/isaac_ros_common
./scripts/run_dev.sh  /home/$USER/Documents/ACC_Development/Development
colcon build
source install/setup.bash
ros2 launch qcar2_nodes qcar2_virtual_launch.py
```
### 4 Open a new terminal to run our code
```bash
docker exec -it isaac_ros_dev-x86_64-container bash
source source install/setup.bash
cd ~/Documents/Buffalo-Bulls-Quanser-Competition/Detection
python3 yolo_detection.py
```
