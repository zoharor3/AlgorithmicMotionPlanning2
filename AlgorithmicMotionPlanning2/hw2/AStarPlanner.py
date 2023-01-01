import numpy as np
import heapq


class AStarPlanner(object):
    def __init__(self, planning_env):
        self.planning_env = planning_env
        self.epsilon = 20

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []

    class Node:
        def __init__(self, parent=None, position=None):
            self.parent = parent
            self.position = position

            self.g = 0
            self.h = 0
            self.f = 0

        def __eq__(self, other):
            return np.array_equal(self.position, other.position)

    def plan(self):
        '''
            Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # TODO: Task 4.3
        possible_movements = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        start = tuple(self.planning_env.start)
        end = tuple(self.planning_env.goal)
        start_node = self.Node(None, start)
        start_node.g = start_node.h = start_node.f = 0

        open_dict = {}
        closed_dict = {}
        open_dict[start] = start_node

        open_f_heap = []
        heapq.heappush(open_f_heap, (0, start))

        while len(open_dict) > 0:
            current_point = heapq.heappop(open_f_heap)[1]
            if current_point in closed_dict:
                continue
            closed_dict[current_point] = open_dict[current_point]
            del open_dict[current_point]
            self.expanded_nodes.append(current_point)
            if current_point == end:
                path = []
                current = closed_dict[current_point]
                path_cost = 0
                while current is not None:
                    path.append(np.array(current.position))
                    if current.parent is not None:
                        path_cost += self.planning_env.compute_distance(np.array(current.parent.position),
                                                                        np.array(current.position))
                    current = current.parent
                print("Path cost:")
                print(path_cost)
                print("Number of states expanded")
                print(len(closed_dict))
                return np.array(path[::-1])
            current_node = closed_dict[current_point]
            for new_position in possible_movements:  # Adjacent squares
                # Get node position
                node_position = [current_point[0] + new_position[0], current_point[1] + new_position[1]]

                # Make sure within range
                if not self.planning_env.state_validity_checker(node_position):
                    continue
                # Make sure walkable terrain
                if not self.planning_env.edge_validity_checker(current_point, node_position):
                    continue

                # Create new node
                child = self.Node(current_node, tuple(node_position))
                if child.position in closed_dict:
                    continue
                child.g = child.parent.g + self.planning_env.compute_distance(np.array(child.parent.position), node_position)
                child.h = self.planning_env.compute_heuristic(node_position)
                child.f = child.g + self.epsilon * child.h
                if child.position in open_dict:
                    if child.g > open_dict[child.position].g:
                        continue
                    else:
                        del open_dict[child.position]
                open_dict[child.position] = child
                heapq.heappush(open_f_heap, (child.f, child.position))
        return None

    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes
