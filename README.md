# ASPIDA robot - ASPiDA ESPA PROJECT

### Prerequisites:
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
$ sudo rosdep init
$ rosdep update
$ rosdep install --from-paths src --ignore-src -r -y
```
```
$ catkin_make
```
## Start simulation

Launch gazebo model :
```
roslaunch ridgeback_gazebo ridgeback_world.launch
```

Launch the navigation :
```
roslaunch ridgeback_navigation odom_navigation_demo.launch
```
Visualize the rviz configuration:
```
roslaunch ridgeback_viz view_robot.launch config:=navigation
```
### Making a Map

Launch the gmapping:
```
roslaunch ridgeback_navigation gmapping_demo.launch
```
Launch rviz :
```
roslaunch ridgeback_viz view_robot.launch config:=gmapping
```

Save the produced map using map_saver:
```
rosrun map_server map_saver -f mymap
```
This will create a mymap.yaml and mymap.pgm file in your current directory.

### Navigation With a Map

Run the AMCL:
```
roslaunch ridgeback_navigation amcl_demo.launch [map_file:=/path/to/my/map.yaml]
```
Run Rviz:
```
roslaunch ridgeback_viz view_robot.launch config:=localization
```

