import numpy as np
import time
import statistics
import math
from RRTTree import RRTTree

class TaskInspectionPlanner(object):

    def __init__(self, planning_env, coverage):

        # set environment
        self.planning_env = planning_env


        # set search params
        self.coverage = coverage

        self.robot = self.planning_env.insp_robot
        self.goal_prob = 0.05
        self.POI_center = []


    def time_plan_split(self):
        gripper_plan = self.planning_env.gripper_plan
        end_gripper_points_plan = []
        for gripper in gripper_plan:
            end_gripper_points_plan.append(self.robot.compute_forward_kinematics(gripper)[-1])
        delta_distance_end_gripper_plan = []
        for i in range(len(end_gripper_points_plan)-1):
            delta_distance_end_gripper_plan.append(math.dist(end_gripper_points_plan[i],end_gripper_points_plan[i+1]))
        mean = statistics.mean(delta_distance_end_gripper_plan)
        stdv = statistics.stdev(delta_distance_end_gripper_plan,mean)
        split_time = [0]
        for i, delta_dist in enumerate(delta_distance_end_gripper_plan):
            if delta_dist - mean > stdv:
                split_time.append(i)
        return split_time


    def get_config_next_to_POI(self, num_of_iter_no_new_points):
        factor0 = 1 - np.exp(-0.001 * num_of_iter_no_new_points)
        factor1 = 1 - np.exp(-0.005 * num_of_iter_no_new_points)
        factor2 = 1 - np.exp(-0.01 * num_of_iter_no_new_points)
        factor3 = 1 - np.exp(-0.1 * num_of_iter_no_new_points)
        cand_config = np.array([np.random.uniform(self.POI_center[0] - np.pi / 32 * factor0,
                                                  self.POI_center[0] + np.pi / 32 * factor0),
                                np.random.uniform(self.POI_center[1] - np.pi / 16 * factor1,
                                                  self.POI_center[1] + np.pi / 16 * factor1),
                                np.random.uniform(self.POI_center[2] - np.pi / 2 * factor2,
                                                  self.POI_center[2] + np.pi / 2 * factor2),
                                np.random.uniform(self.POI_center[3] - np.pi / 2 * factor3,
                                                  self.POI_center[3] + np.pi / 2 * factor3)])
        return cand_config

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        eta = 0.6

        distance = self.planning_env.insp_robot.compute_distance(near_config, rand_config)
        direction = (rand_config - near_config) * 1 / distance
        new_cand_config = near_config + eta * distance * direction
        return new_cand_config

    def compute_union_of_points(self, points1, points2):
        '''
        Compute a union of two sets of inpection points.
        @param points1 list of inspected points.
        @param points2 list of inspected points.
        '''
        if points1 is None or len(points1) == 0:
            points = points2
        elif points2 is None or len(points2) == 0:
            points = points1
        else:
            points = points1 or points2
        if points is None:
            return None
        else:
            return points


    def plan_part(self,time_start,time_end):
        #TODO use compute_inspected_timestamps_for_edge both in begining and in during search
        #TODO add to rrt timestamps

        tree = RRTTree(self.planning_env, task="ip")
        plan_configs, plan_timestamps = [], []
        cand_config = self.POI_center
        goal_bias_counter = 0
        num_of_iter_no_new_points = 100

        inspected_points_from_start_config = []  # TODO CHECK start point
        tree.add_vertex(self.planning_env.inspector_start, inspected_points_from_start_config, time_stamp=0)
        while tree.max_coverage < self.coverage:
            num_of_iter_no_new_points += 1
            goal_bias_counter += 1
            goal_prob = np.exp(-0.01 * num_of_iter_no_new_points) + self.goal_prob
            [limitMin, limitMax] = [-np.pi, np.pi]
            if goal_bias_counter < 1 / goal_prob:
                cand_config = np.array(
                    [np.random.uniform(limitMin, 0.5*limitMin),  # i can change this to np.pi/2 if always robot origin is (0,0)
                     np.random.uniform(limitMin, limitMax),
                     np.random.uniform(limitMin, limitMax),
                     np.random.uniform(limitMin, limitMax)])
                cand_time_stamp = np.random.randint(time_end-time_start+2)+time_start
            else:
                if len(self.POI_center) != 0:
                    cand_config = self.get_config_next_to_POI(num_of_iter_no_new_points)
                goal_bias_counter = 0
            [nearest_id, nearest_config, nearst_time_stamp] = tree.get_nearest_config(config=cand_config, time_stamp=cand_time_stamp)
            if nearst_time_stamp >= cand_time_stamp:
                continue

            cand_config_extend = self.extend(nearest_config, cand_config)
            if self.planning_env.config_validity_checker(cand_config_extend, "inspect"):
                if self.planning_env.edge_validity_checker(config1=cand_config_extend, config2=nearest_config,robot_type = "inspect"):
                    points_so_far = tree.vertices[nearest_id].inspected_points
                    current_config_inspected_points = self.planning_env.compute_inspected_timestamps_for_edge(nearest_config,
                                                                                                              cand_config_extend,
                                                                                                              timestamp1=nearst_time_stamp,
                                                                                                              timestamp2=cand_time_stamp)
                    if current_config_inspected_points is not None:
                        self.POI_center = cand_config_extend
                    inspected_points = self.compute_union_of_points(points_so_far,
                                                                    current_config_inspected_points)
                    cand_config_extend_id = tree.add_vertex(cand_config_extend, inspected_points=inspected_points, time_stamp=cand_time_stamp)
                    tree.add_edge(nearest_id,
                                       cand_config_extend_id,
                                       self.robot.compute_distance(nearest_config, cand_config_extend),
                                  self.robot.compute_distance_with_time(nearest_config, cand_config_extend, nearst_time_stamp, cand_time_stamp))
                    if tree.does_coverage_increased:
                        num_of_iter_no_new_points = 0

        curr_ver_id = tree.max_coverage_id
        plan_configs.append(tree.vertices[curr_ver_id].config)
        plan_timestamps.append(tree.vertices[curr_ver_id].time_stamp)
        while curr_ver_id != 0:
            curr_ver_id = tree.edges[curr_ver_id]
            plan_configs.append(tree.vertices[curr_ver_id].config)
            plan_timestamps.append(tree.vertices[curr_ver_id].time_stamp)
        plan_configs.reverse()
        plan_timestamps.reverse()
        return plan_configs, plan_timestamps, tree.max_coverage

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan_configs, plan_timestamps = [], []

        # TODO: implement you planner here
        # Your stopping condition should look like this:
        # while coverage < self.coverage:
        time_point_split = self.time_plan_split()
        start_initial_time = 0
        current_coverage = 0
        total_points = len(self.planning_env.gripper_plan)
        # for end_time in time_point_split:
        #     partial_plan_configs, partial_plan_timestamps, partial_coverage = self.plan_part(start_initial_time,end_time)
        #     plan_configs.extend(partial_plan_configs)
        #     plan_timestamps.extend(partial_plan_timestamps)
        #     current_coverage += (end_time-start_initial_time)* partial_coverage / total_points
        #     start_initial_time = end_time + 1

        if current_coverage < self.coverage:
            temp_plan_configs, temp_plan_timestamps, temp_current_coverage = self.plan_part(0, total_points)
            if temp_current_coverage > current_coverage:
                plan_configs, plan_timestamps, current_coverage = temp_plan_configs, temp_plan_timestamps, temp_current_coverage

        # store total path cost and time
        path_cost = self.compute_cost(plan_configs)
        computation_time = time.time()-start_time

        filename = "results_prob_" + str(self.coverage) + ".txt"
        with open(filename, "a") as f:
            f.write('Run at {:.2f}'.format(start_time))
            f.write('Total cost of path: {:.2f}'.format(path_cost))
            f.write("\n")
            f.write('Total Computation time: {:.2f}'.format(computation_time))
            f.write("\n")
            f.write('Total Coverage: {:.2f}'.format(current_coverage))
            f.write("\n")

        return np.array(plan_configs), np.array(plan_timestamps), current_coverage, path_cost, computation_time

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # compute cost of a given path
        plan_cost = 0.0
        for i in range(len(plan)-1):
            plan_cost += self.planning_env.insp_robot.compute_distance(plan[i], plan[i+1])
        return plan_cost