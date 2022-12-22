import numpy as np
from RRTTree import RRTTree
import time

class RRTPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # TODO: Task 4.4
        tree = RRTTree
        while not tree.is_goal_exists(tree, self.planning_env.goal):
            cand_stat = [np.random.uniform(self.planning_env.xlimit), np.random.uniform(self.planning_env.ylimit)]
            if self.planning_env.state_validity_checker(cand_stat):
                nearest_state = tree.get_nearest_state(cand_stat)
                if self.planning_env.edge_validity_checker(cand_stat, nearest_state):
                    tree.add_edge(tree.get_idx_for_state(nearest_state), tree.get_idx_for_state(cand_stat),
                                  self.planning_env.compute_distance(nearest_state, cand_stat))

        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 4.4

        pass

    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''
        # TODO: Task 4.4
        if self.ext_mode == 'E1':
            pass
        else:
            pass
        pass