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
        goal_bias_counter = 0
        tree = RRTTree
        self.tree.add_vertex(self.planning_env.start)
        while not tree.is_goal_exists(self.tree, self.planning_env.goal):
            goal_bias_counter += 1
            [xLimitMin, xLimitMax] = self.planning_env.xlimit
            [yLimitMin, yLimitMax] = self.planning_env.ylimit
            if goal_bias_counter < 1/self.goal_prob:
                cand_state = np.array([np.random.uniform(xLimitMin,xLimitMax), np.random.uniform(yLimitMin, yLimitMax)])
            else:
                cand_state = self.planning_env.goal
                goal_bias_counter = 0
            [nearest_id, nearest_state] = tree.get_nearest_state(self=self.tree, state=cand_state)
            cand_state_extend = self.extend(nearest_state, cand_state)
            if self.planning_env.state_validity_checker(cand_state_extend):
                if self.planning_env.edge_validity_checker(state1=cand_state_extend, state2=nearest_state):
                    tree.add_vertex(self.tree, cand_state_extend)
                    tree.add_edge(self.tree, tree.get_idx_for_state(self.tree, nearest_state),
                                  tree.get_idx_for_state(self.tree, cand_state_extend),
                                  self.planning_env.compute_distance(nearest_state, cand_state_extend))
        # print total path cost and time
        plan.append(self.planning_env.goal)
        i = 0
        while tree.get_idx_for_state(self.tree, plan[i]) !=  tree.get_idx_for_state(self.tree, self.planning_env.start):
            curr_ver_id = tree.get_idx_for_state(self.tree, plan[i])
            next_ver_id = self.tree.edges[curr_ver_id]
            i += 1
            plan.append(self.tree.vertices[next_ver_id].state)
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 4.4
        tot_cost = 0
        for i in range(len(plan)-1):
            tot_cost += self.planning_env.compute_distance(plan[i], plan[i+1])
        return tot_cost

    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''
        # TODO: Task 4.4
        eta = 0.6
        if self.ext_mode == 'E1':
            return rand_state
        else:
            distance = self.planning_env.compute_distance(near_state, rand_state)
            direction = (rand_state-near_state)*1/distance
            new_cand_state = near_state+eta*distance*direction
        return new_cand_state