#! /usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

rospy.init_node('actor_pose_pub')

pose_pub=rospy.Publisher ('/bus', PoseStamped, queue_size = 10)

rospy.wait_for_service ('/gazebo/get_model_state')
get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

pose=PoseStamped()
header = Header()
header.frame_id='/map'

model = GetModelStateRequest()
model.model_name='bus'

r = rospy.Rate(100)

while not rospy.is_shutdown():
    result = get_model_srv(model)

    pose.pose = result.pose
    # odom.twist.twist = result.twist

    header.stamp = rospy.Time.now()
    pose.header = header

    pose_pub.publish (pose)

    r.sleep()
