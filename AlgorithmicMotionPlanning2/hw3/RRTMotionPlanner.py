import numpy as np
from RRTTree import RRTTree
import time
import Robot as Robot

class RRTMotionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)
        self.robot = Robot.Robot
        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.found_plan = False

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []
        # TODO: Task 2.3


        goal_bias_counter = 0
        self.tree.add_vertex(self.planning_env.start)
        while not self.tree.is_goal_exists(self.planning_env.goal):
            goal_bias_counter += 1
            [limitMin, limitMax] = [-np.pi, np.pi]
            if goal_bias_counter < 1/self.goal_prob:
                cand_config = np.array([np.random.uniform(limitMin,limitMax), # i can change this to np.pi/4 if always robot origin is (0,0)
                                        np.random.uniform(limitMin,limitMax),
                                        np.random.uniform(limitMin,limitMax),
                                        np.random.uniform(limitMin,limitMax)])
            else:
                cand_config = self.planning_env.goal
                goal_bias_counter = 0
            [nearest_id, nearest_config] = self.tree.get_nearest_config(config=cand_config)
            cand_config_extend = self.extend(nearest_config, cand_config)
            if self.planning_env.config_validity_checker(cand_config_extend):
                if self.planning_env.edge_validity_checker(config1=cand_config_extend, config2=nearest_config):
                    cand_config_extend_id = self.tree.add_vertex(cand_config_extend)
                    self.tree.add_edge(nearest_id,
                                   cand_config_extend_id,
                                   self.robot.compute_distance(self.robot, nearest_config, cand_config_extend))


        curr_ver_id = len(self.tree.vertices)-1
        plan.append(self.tree.vertices[curr_ver_id].config)
        while curr_ver_id != 0:
            curr_ver_id = self.tree.edges[curr_ver_id]
            plan.append(self.tree.vertices[curr_ver_id].config)
        plan.reverse()

        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        with open("results_prob_0.05.txt", "a") as f:
            f.write('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
            f.write("\n")
            f.write('Total time: {:.2f}'.format(time.time()-start_time))
            f.write("\n")

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.3
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
        # TODO: Task 2.3
        eta = 0.6
        if self.ext_mode == 'E1':
            return rand_config
        else:
            distance = self.planning_env.compute_distance(near_config, rand_config)
            direction = (rand_config-near_config)*1/distance
            new_cand_config = near_config+eta*distance*direction
        return new_cand_config
    