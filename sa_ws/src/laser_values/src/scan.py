#! /usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan

"""
def callback(msg):
    print(len(msg.ranges))
"""

def callback(msg):
    """
    # values at 0 degree
    print(msg.ranges[0])
    # values at 90 degree
    print(msg.ranges[360])
    # values at 180 degree
    print(msg.ranges[719])"""

    """
    print('angle min, max, increment')
    print(msg.angle_min)
    print(msg.angle_max)
    print(msg.angle_increment)
    print('range_min and max')
    print(msg.range_min)
    print(msg.range_max)
    """
    print(msg.ranges[0])



rospy.init_node('scan_values')
sub = rospy.Subscriber('/scan', LaserScan, callback)
rospy.spin()