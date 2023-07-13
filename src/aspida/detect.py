#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose,PoseWithCovarianceStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from visualization_msgs.msg import Marker

from move_base_msgs.msg import MoveBaseAction,MoveBaseGoal
from actionlib.simple_action_client import SimpleActionClient

bridge = CvBridge()

# Rviz marker publish
# rospy.init_node('rviz_marker')
marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 10)
marker = Marker()

goal=MoveBaseGoal()
ac= SimpleActionClient('move_base',MoveBaseAction)
received_messages = {
    'image': None,
    'depth': None,
    'pose': None,
    'depth_K':None,
    'color_K':None,
    'camera_info_color':None,
    'camera_info_depth':None
}



# Callback function for processing the received image
def image_callback(msg):
    # Convert the ROS image message to OpenCV format
    cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
    received_messages['image']=cv_image
    # Process the image (example: display it)
    # cv2.imshow("Camera Image", cv_image)
    # cv2.waitKey(1)

def depth_image_callback(msg):

    # Convert the ROS image message to OpenCV format
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    cv_image = np.array(depth_image, dtype=np.float32)

    # cv_image = bridge.imgmsg_to_cv2(msg, "8UC1")
    received_messages['depth']=cv_image
    # Process the image (example: display it)
    # cv2.imshow("Camera Image", cv_image)
    # cv2.waitKey(1)

def pose_callback(msg):
    # Access position information
    position = msg.pose.pose.position
    x = position.x
    y = position.y
    z = position.z
    t=np.asarray([x,y,z])
    # Access orientation information
    orientation = msg.pose.pose.orientation
    qx = orientation.x
    qy = orientation.y
    qz = orientation.z
    qw = orientation.w
    q=np.asarray([qw,qx,qy,qz])
    # Print the pose information
    print("Position (x, y, z): ({}, {}, {})".format(x, y, z))
    print("Orientation (qx, qy, qz, qw): ({}, {}, {}, {})".format(qx, qy, qz, qw))
    print("----------------------")
    received_messages['pose']=[t,q]
    return(t,q)

def camera_params_color(cameraInfo):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    # _intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none     
    # _intrinsics.coeffs = [i for i in cameraInfo.D]
    received_messages['camera_info_color']=_intrinsics
    # rs.pyrealsense2.intrinsics=received_messages['camera_info_color']

def camera_params_depth(cameraInfo):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    # _intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none     
    # _intrinsics.coeffs = [i for i in cameraInfo.D]
    received_messages['camera_info_depth']=_intrinsics
    # rs.pyrealsense2.intrinsics=received_messages['camera_info_depth']  

def subscribe_to_topics():
    # Initialize the ROS node
    rospy.init_node("camera_listener", anonymous=True)

    # Subscribe to the camera topic
    # rospy.Subscriber("/d435/depth/image_raw", Image, depth_image_callback)
    # rospy.Subscriber("/d435/color/image_raw", Image, image_callback)
    # rospy.Subscriber("/d435/color/camera_info", CameraInfo,camera_params_color)
    # rospy.Subscriber("/d435/depth/camera_info", CameraInfo,camera_params_depth)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_image_callback)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.Subscriber("/camera/color/camera_info", CameraInfo,camera_params_color)
    rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo,camera_params_depth)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, pose_callback)



    # Spin the ROS node to receive callbacks
    # rospy.spin()

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        process_messages()
        
        rate.sleep()



def process_messages():
    if received_messages['image'] is None or received_messages['depth'] is None or not received_messages['pose'] is None: return(0)
    image=received_messages['image']
    xyz=get_pose(image)
    if xyz is None: 
        print('Person not found!')
    else:
        pass
        toRviz(xyz)
        movebase_client(xyz)
        

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ptrack=[]
def get_pose(image):
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results=pose.process(image)
    if results.pose_landmarks is not None:
        landmarks=results.pose_landmarks.landmark
        xyz=[np.asarray([l.x,l.y,l.z]) for l in landmarks]
        #transform to true size
        h,w,_=image.shape
        xyz=(np.asarray(xyz)*np.asarray([w,h,1]))[:,:2]

        #remove landmarks that are outside of the image 
        mask = (xyz[:, 0] >= 0) & (xyz[:, 0] <= w) & (xyz[:, 1] >= 0) & (xyz[:, 1] <= h)
        xyz = xyz[mask]

        depth=received_messages['depth'][xyz[:,1].astype(int),[xyz[:,0].astype(int)]]/1000
        depth_m=np.median(depth)
        xyz_m=np.median(xyz,axis=0)
        p3d=rs.rs2_deproject_pixel_to_point(received_messages['camera_info_depth'],xyz_m[:2],depth_m)
        # ptrack.append(p3d)
        # i=0
        # depth=received_messages['depth'][xyz[i,0].astype(int),[xyz[i,1].astype(int)]]/1000
        # p3d=rs.rs2_deproject_pixel_to_point(received_messages['camera_info_color'],xyz[i,:2].astype(int),depth)
        return(p3d)
    else: 
        return(None)



def movebase_client(target):
    client = SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = target[0]
    goal.target_pose.pose.position.y = target[1]
    goal.target_pose.pose.orientation.w = 1.0
    
    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        return client.get_result()

def toRviz(xyz):
    marker.header.frame_id = "camera_color_optical_frame"#"/map"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = 2
    marker.id = 0

    # Set the scale of the marker
    marker.scale.x = .5
    marker.scale.y = .5
    marker.scale.z = .5

    # Set the color
    marker.color.r = 0.0
    marker.color.g = 100.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Set the pose of the marker
    marker.pose.position.x = xyz[0]
    marker.pose.position.y = xyz[1]
    marker.pose.position.z = xyz[2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker_pub.publish(marker)

# def convertTo3D(K,pts,depth):
#     # pts in pixel, not normalized
#     pcd = []
    
#     for i,j in pts:
#            z = depth[i,j]
#            x = (j - CX_DEPTH) * z / FX_DEPTH
#            y = (i - CY_DEPTH) * z / FY_DEPTH
#            pcd.append([x, y, z])
#     return(pcd)


# frame = cv2.imread('catkin_ws/src/myscripts/sample.jpg')
# image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# results=pose.process(image)
# mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)










if __name__ == "__main__":
    try:
        subscribe_to_topics()
    except rospy.ROSInterruptException:
        pass