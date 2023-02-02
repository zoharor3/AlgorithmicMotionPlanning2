import numpy as np
from RRTTree import RRTTree
import time

class RRTMotionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # TODO: Task 2.3
        
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.3
        tot_cost = 0
        for i in range(len(plan)-1):
            tot_cost += self.planning_env.compute_distance(plan[i], plan[i+1])
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
            new_cand_state = near_config+eta*distance*direction
        return new_cand_state
    