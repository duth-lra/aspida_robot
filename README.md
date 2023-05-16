# ASPIDA robot for ASPiDA ESPA PROJECT

Prerequisites:
Install Ubuntu 20.04 - ROS noetic 

Create a catkin workspace following the intructions: [catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)

Copy and paste in the ~/catkin_ws/src the following command:
```
$ cd ~/catkin_ws/src/
```

```
$ git clone https://github.com/duth-lra/ridgeback
$ git clone https://github.com/duth-lra/ridgeback_simulator
$ git clone https://github.com/duth-lra/ridgeback_robot
$ git clone https://github.com/duth-lra/ridgeback_baxter
$ git clone https://github.com/ros-planning/navigation
$ git clone https://github.com/nilseuropa/realsense_ros_gazebo
```
```
$ cd ..
```
```
$ rosdep install --from-paths src --ignore-src -r -y
```
