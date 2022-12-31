import numpy as np


class AStarPlanner(object):
    def __init__(self, planning_env):
        self.planning_env = planning_env
        self.epsilon = 10

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
        start = self.planning_env.start
        end = self.planning_env.goal
        start_node = self.Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = self.Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)
            self.expanded_nodes.append(current_node.position)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(np.array(current.position))
                    current = current.parent
                return np.array(path[::-1])  # Return reversed path

            # Generate children
            children = []
            for new_position in possible_movements:  # Adjacent squares
                # Get node position
                node_position = [current_node.position[0] + new_position[0], current_node.position[1] + new_position[1]]

                # Make sure within range
                if not self.planning_env.state_validity_checker(node_position):
                    continue

                # Make sure walkable terrain
                if not self.planning_env.edge_validity_checker(current_node.position, node_position):
                    continue

                # Create new node
                new_node = self.Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = child.parent.g + self.planning_env.compute_distance(child.parent.position, child.position)
                child.h = self.planning_env.compute_heuristic(child.position)
                child.f = child.g + self.epsilon * child.h

                # Child is already in the open list
                pop_list = []
                for index, open_node in enumerate(open_list):
                    if child == open_node:
                        if child.g > open_node.g:
                            continue
                        else:
                            pop_list.append(index)
                for index in sorted(pop_list, reverse=True):
                    del open_list[index]
                # Add the child to the open list
                open_list.append(child)
        return None

    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes
