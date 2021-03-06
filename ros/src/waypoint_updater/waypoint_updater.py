#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
import numpy as np

from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        #rospy.spin()
        self.pose = None
        #self.base_waypoints = None
        self.stopline_wp_idx = -1
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.loop()

    def loop(self):
        print "-----------------------"
        print "<<<<<<Enter in the loop"
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            #print "self.base_waypoints: ",self.base_waypoints,self.pose
            if self.pose and self.base_lane:
                #print "he entrado"
                closest_waypoint_idx = self.get_closest_waypoint_id()
                self.publish_waypoints(closest_waypoint_idx)
	    rate.sleep()
    def get_closest_waypoint_id(self):
        x=self.pose.pose.position.x
        y=self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        #check ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        #hyperplane
        cl_vect = np.array([closest_coord][0])
        prev_vect = np.array([prev_coord][0])
        pos_vect = np.array([x,y])
        #print "cl_vect: ",cl_vect
        #print "prev_vect: ",prev_vect
        #print "pos_vect: ",pos_vect

        val = np.dot(cl_vect - prev_vect,pos_vect - cl_vect)
        if val>0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self,closest_idx):
        #print ">>>>>>> Publish waypoints"
        #lane = Lane()
        #lane.header = self.base_waypoints.header
        #lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx+LOOKAHEAD_WPS]
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
	#print ">>>>>>>> GENERATE LANE"
        lane = Lane()
        closest_idx = self.get_closest_waypoint_id()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        #rospy.logwarn("closest_idx: {0} | farthest_idx: {1}".format(closest_idx, farthest_idx))
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        #elif false:
        else:
            rospy.logwarn("stopline_wp_idx: {0} | farthest_idx: {1}".format(self.stopline_wp_idx, farthest_idx))
            lane.waypoints = self.decelerate_waypoints(base_waypoints,closest_idx)
        return lane
    def decelerate_waypoints(self,waypoints,closest_idx):
        temp = []
	#rospy.logwarn("self.stopline_wp_idx {0} | closest_idx {1}".format(self.stopline_wp_idx,closest_idx))
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            stop_idx = max(self.stopline_wp_idx - closest_idx -2, 0)
	    #rospy.logwarn("i: {0}, stop_idx: {1},len(waypoints):{2},closest_idx: {3}".format(i,stop_idx,len(waypoints),closest_idx))
            dist = self.distance(waypoints,i,stop_idx)

            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.
            p.twist.twist.linear.x = min(vel,wp.twist.twist.linear.x)
            temp.append(p)
        return temp
    def pose_cb(self, msg):
        # TODO: Implement

        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        print "waypoints cb"
        #print "waypoints: ",waypoints
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d) 

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
	#print ">>>>>> traffic cb"
	#rospy.logwarn("traffic_cb {0}".format(msg.data))
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
