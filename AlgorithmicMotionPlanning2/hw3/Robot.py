import itertools
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from shapely.geometry import Point, LineString, Polygon

class Robot(object):
    
    def __init__(self):

        # define robot properties
        self.links = np.array([80.0,70.0,40.0,40.0])
        self.dim = len(self.links)

        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi/3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        # TODO: Task 2.2
        return np.linalg.norm(prev_config - next_config)

    def compute_forward_kinematics(self, given_config):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        # TODO: Task 2.2
        links_location = np.zeros((4, 2))
        total_angle = 0
        for i, link in enumerate(self.links):
            if i == 0:
                total_angle += given_config[0]
                links_location[0, 0] = link * np.cos(total_angle)
                links_location[0, 1] = link * np.sin(total_angle)
            else:
                total_angle += given_config[i]
                links_location[i, 0] = links_location[i-1, 0] + link * np.cos(total_angle)
                links_location[i, 1] = links_location[i-1, 1] + link * np.sin(total_angle)
        return links_location

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1,len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        if link_angle + given_angle > np.pi:
            return link_angle + given_angle - 2*np.pi
        elif link_angle + given_angle < -np.pi:
            return link_angle + given_angle + 2*np.pi
        else:
            return link_angle + given_angle
        
    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        # TODO: Task 2.2
        poly = LineString(robot_positions)
        for i in range(robot_positions.shape[0] - 1):
            line = LineString(robot_positions[i:i + 2, :])
            if poly.intersects(line) and not line.within(poly):
                return False
        return True
    