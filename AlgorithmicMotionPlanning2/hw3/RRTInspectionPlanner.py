import numpy as np
from RRTTree import RRTTree
import time
import Robot as Robot

class RRTInspectionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env, task="ip")
        self.robot = Robot.Robot


        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage
        self.found_plan = False
        self.POI_center = []
        # self.POI_tolerance = np.pi/4


    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # TODO: Task 2.4

        # your stopping condition should look like this: 
        # while self.tree.max_coverage < self.coverage:
        goal_bias_counter = 0
        num_of_iter_no_new_points = 100
        inspected_points_from_start_config = self.planning_env.get_inspected_points(config=self.planning_env.start)
        self.tree.add_vertex(self.planning_env.start, inspected_points_from_start_config)
        while self.tree.max_coverage < self.coverage:
            num_of_iter_no_new_points += 1
            goal_bias_counter += 1
            goal_prob = np.exp(-0.01*num_of_iter_no_new_points)+self.goal_prob
            [limitMin, limitMax] = [-np.pi, np.pi]
            if goal_bias_counter < 1/goal_prob:
                cand_config = np.array([np.random.uniform(0, limitMax/2), # i can change this to np.pi/2 if always robot origin is (0,0)
                                        np.random.uniform(limitMin, limitMax),
                                        np.random.uniform(limitMin, limitMax),
                                        np.random.uniform(limitMin, limitMax)])
            else:
                if len(self.POI_center) != 0:
                    cand_config = self.get_config_next_to_POI(num_of_iter_no_new_points)
                goal_bias_counter = 0
            [nearest_id, nearest_config] = self.tree.get_nearest_config(config=cand_config)
            cand_config_extend = self.extend(nearest_config, cand_config)
            if self.planning_env.config_validity_checker(cand_config_extend):
                if self.planning_env.edge_validity_checker(config1=cand_config_extend, config2=nearest_config):
                    points_so_far = self.tree.vertices[nearest_id].inspected_points
                    current_config_inspected_points = self.planning_env.get_inspected_points(config=cand_config_extend)
                    if len(current_config_inspected_points) != 0:
                        self.POI_center = cand_config_extend
                    inspected_points = self.planning_env.compute_union_of_points(points_so_far, current_config_inspected_points)
                    cand_config_extend_id = self.tree.add_vertex(cand_config_extend, inspected_points=inspected_points)
                    self.tree.add_edge(nearest_id,
                                  cand_config_extend_id,
                                  self.robot.compute_distance(self.robot, nearest_config, cand_config_extend))
                    if self.tree.does_coverage_increased:
                        num_of_iter_no_new_points = 0

        curr_ver_id = self.tree.max_coverage_id
        plan.append(self.tree.vertices[curr_ver_id].config)
        while curr_ver_id != 0:
            curr_ver_id = self.tree.edges[curr_ver_id]
            plan.append(self.tree.vertices[curr_ver_id].config)
        plan.reverse()

        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))
        with open("results_coverage_0.75.txt", "a") as f:
            f.write('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
            f.write("\n")
            f.write('Total time: {:.2f}'.format(time.time()-start_time))
            f.write("\n")
        return np.array(plan)

    def get_config_next_to_POI(self, num_of_iter_no_new_points):
        factor0 = 1-np.exp(-0.001*num_of_iter_no_new_points)
        factor1 = 1-np.exp(-0.005*num_of_iter_no_new_points)
        factor2 = 1-np.exp(-0.01*num_of_iter_no_new_points)
        factor3 = 1-np.exp(-0.1*num_of_iter_no_new_points)
        cand_config = np.array([np.random.uniform(self.POI_center[0] - np.pi/32*factor0,
                                                  self.POI_center[0] + np.pi/32*factor0),
                                np.random.uniform(self.POI_center[1] - np.pi/16*factor1,
                                                  self.POI_center[1] + np.pi/16*factor1),
                                np.random.uniform(self.POI_center[2] - np.pi/2*factor2,
                                                  self.POI_center[2] + np.pi/2*factor2),
                                np.random.uniform(self.POI_center[3] - np.pi/2*factor3,
                                                  self.POI_center[3] + np.pi/2*factor3)])
        return cand_config

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.4
        tot_cost = 0
        for i in range(len(plan)-1):
            tot_cost += self.robot.compute_distance(self, plan[i], plan[i+1])
        return tot_cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: Task 2.4
        eta = 0.6
        if self.ext_mode == 'E1':
            return rand_config
        else:
            distance = self.planning_env.compute_distance(near_config, rand_config)
            direction = (rand_config-near_config)*1/distance
            new_cand_config = near_config+eta*distance*direction
        return new_cand_config

    