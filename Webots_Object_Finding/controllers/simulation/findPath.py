'''
Tamer Abousoud

Find Path through World
---
Script to find best path between points in the simulation.
Uses Dijkstra's shortest distance algorithm and applies some
simple transformations to provide simple robot instructions.
'''


# import robotControl
from worldMap import *

import math
import heapq as hp
from bisect import bisect
from itertools import product
from collections import defaultdict, deque, namedtuple
from scipy.spatial import distance



def simplify_path(path, tol=0.01):
    ''' 
    Simplify a path to remove redundant vertices
    ---
    path: list of vertices in path
    tol: relative tolerance for comparing slope
    '''
    redundant = []
    
    for i in range(1, len(path)-1):

        x, x_prev, x_next = path[i][0], path[i-1][0], path[i+1][0]
        z, z_prev, z_next = path[i][1], path[i-1][1], path[i+1][1]
        
        dx_prev, dx_next = x-x_prev, x_next-x
        dz_prev, dz_next = z-z_prev, z_next-z

        # Simple case:
        # x or z value is constant
        if (dx_prev == 0 and dx_next == 0) or (dz_prev == 0 and dz_next == 0):
            redundant.append(path[i])
        # Diagonal case:
        # slope (delta_z/delta_x) is constant
        elif math.isclose(dx_prev, dx_next, rel_tol=0.01) and math.isclose(dz_prev, dz_next,\
            rel_tol=0.01):
            redundant.append(path[i])

    return [p for p in path if not p in redundant]


def path_to_sequence(path):
    ''' 
    Convert a path to sequence of straight lines and angles for input
    to the robot controllers
    ---
    path: a simplified path (run `simplify_path()` first)
    
    Returns angle, distance to move robot. Positive angles are counter-clockwise.
    '''
    path_segments = [(path[i-1], path[i]) for i in range(1, len(path))]
    travel_distances = [distance.euclidean(s[0], s[1]) for s in path_segments]

    turn_angles = []

    for segment, ds in zip(path_segments, travel_distances):

        x1, z1, x2, z2 = segment[0][0], segment[0][1], segment[1][0], segment[1][1]

        if segment == path_segments[0]:
            # Get rotated angle at start
            dx, dz = x2 - x1, z2 - z1
            # 1st angle to turn robot towards path start
            theta = math.acos(abs(dz)/ds)
            if dx > 0:  # determine if pt is left/right of robot
                theta = -theta
            if dz > 0:  # determine if pt is in front/behind robot
                theta = math.pi - theta
            # Define robot starting angle
            start_angle = math.degrees(theta)
            # First turn angle is zero after orienting robot at start
            turn_angle = 0

        else:
            # Calculate transformed coordinates (x`, z`)
            x1_p = x1 * math.cos(theta) - z1 * math.sin(theta)
            z1_p = x1 * math.sin(theta) + z1 * math.cos(theta)
            x2_p = x2 * math.cos(theta) - z2 * math.sin(theta)
            z2_p = x2 * math.sin(theta) + z2 * math.cos(theta)
            dx, dz = x2_p - x1_p, z2_p - z1_p
            # Turn angle relative to transformed coords
            turn_angle = math.acos(abs(dz)/ds)
            if dx > 0:  # determine if pt is left/right of robot
                turn_angle = -turn_angle
            if dz > 0:  # determine if pt is in front/behind robot
                turn_angle = math.pi - turn_angle
            # Transform coordinates as robot moves
            theta = turn_angle + theta
            # print('theta', round(math.degrees(theta), 3))

        turn_angles.append(round(math.degrees(turn_angle), 3))

    # starting_angle: angle to align robot with path relative to initial orientation
    # instructions: (angle, distance) to turn and move robot for each segment
    sequence = {'starting_angle': start_angle, 
                'instructions': list(zip(*(turn_angles, travel_distances)))}

    return sequence


# ========================================================================================== #


class locationUnreachableException(Exception):
    '''Returns error if node cannot be reached'''
    pass



class worldGraph:
    ''' 
    Structures the world into a graph. 
    Used to find the shortest path between locations on the map. 
    '''

    def __init__(self, worldMap):
        ''' 
        worldMap: instance of `worldMap` object for creating the graph
        NOTE:
        worldMap.validate_nodes() should be run prior to using this class
        '''
        self._all_nodes = worldMap.grid
        self._valid_nodes = worldMap.valid_nodes
        self._invalid_nodes = worldMap._invalid_nodes
        # Get node neighbors from `worldMap`
        self._node_neighbors = worldMap.define_node_neighbors()
        # Set up vertices and edges for graph
        self._graph_vertices = self._node_neighbors.keys()
        # self._graph_edges = ()
        # Distances from a mode to its neighbors
        self._node_distances = defaultdict(list)

        # Grid dimensions
        self._world_size = worldMap.size
        self._grid_rows, self._grid_cols = worldMap._nodes.shape[:2]
        self._grid_cell_size = (self._world_size[0]/self._grid_rows, 
                                self._world_size[1]/self._grid_cols)

        # For node searches
        self.__node_array = worldMap._nodes
        self.__nodeGrid = worldMap._worldMap__nodeGrid


    def get_node_distances(self):
        ''' 
        Returns start and end nodes with Euclidean distance between them 
        to be used for calculating path cost.
        '''

        for vtx in self._graph_vertices:
            adjacent_nodes = list(product([vtx], self._node_neighbors.get(vtx)))
            node_dists = [(n[1], distance.euclidean(n[0], n[1])) for n in adjacent_nodes]
            self._node_distances[vtx] = node_dists


    def find_nearest_node(self, point):
        '''
        Find nearest node in `worldMap` graph to `point`
        ---
        point: (x, z) coordinates of point
        '''
        if abs(point[0]) > self._world_size[0]/2 or abs(point[1]) > self._world_size[1]/2:
            raise ValueError(f'Point {point} is outside world boundaries')

        grid_x_coords = self.__nodeGrid[0][1,:]
        grid_z_coords = self.__nodeGrid[1][:,1]

        i_max = bisect(grid_x_coords, point[0])
        i_min = i_max - 1
        j_max = bisect(grid_z_coords, point[1])
        j_min = j_max - 1

        nearest_node_ids = product([i_min, i_max], [j_min, j_max])
        nearest_nodes = [tuple(self.__node_array[idx]) for idx in nearest_node_ids]

        invalid = set(self._invalid_nodes)
        if set(nearest_nodes).issubset(invalid):
            raise ValueError(f'No valid nodes near {point}. Make sure point is not inside an obstacle boundary. You can use worldMap.points_from_map() to select valid points.')

        node_dists = []
        for node in nearest_nodes:
            d = distance.euclidean(point, node) if not node in invalid else np.inf
            node_dists.append((d, node))

        return min(node_dists)[1]


    def find_shortest_path(self, start, end, plot=False):
        '''
        Use Dijkstra's shortest path algorithm to find the shortest
        path between two points in `worldMap`
        ---
        start, end: (x, z) coordinates of start and end points
        '''
        start = self.find_nearest_node(start)
        end = self.find_nearest_node(end)

        inf = np.inf

        # Use node distances for path costs
        # `path_dist` is total cost to reach a node at current step
        path_dist = {nd: 0 if nd == start else inf for nd in self._graph_vertices}
        # Node through which cost is lowest to reach current node
        predecessors = {nd: None for nd in self._graph_vertices}
        shortest_path = deque([end])

        # Find possible paths through a node to its neighbors
        # Add to heap such that smallest path is always first
        possible_paths = [(path_dist.get(nd), nd) for nd in self._graph_vertices]
        hp.heapify(possible_paths)

        while possible_paths:
            current_dist, current_node = hp.heappop(possible_paths)
            # Stop at end point or if node is unreachable from start
            if current_node == end:
                break
            if current_dist == inf:
                raise locationUnreachableException(f'Position {current_node} cannot be reached from {start}.')
            # for neighbor in nodes.get(current_node[1]):
            for neighbor, dist in self._node_distances.get(current_node):
                # Calculate total distance through current node to neighbors
                new_dist = path_dist.get(current_node) + dist
                # If new distance to neighbor is less than current best path
                # update to new distance
                if new_dist < path_dist.get(neighbor):
                    path_dist[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    hp.heappush(possible_paths, (path_dist.get(neighbor), neighbor))

        previous = predecessors.get(end)

        while not previous is None:
            shortest_path.appendleft(previous)
            previous = predecessors.get(previous)

        if plot:
            # Plot the path
            x_valid, z_valid = zip(*self._valid_nodes)
            x_invld, z_invld = zip(*self._invalid_nodes)
            path_x, path_z = zip(*shortest_path)

            _, ax = plt.subplots()

            plt.plot(x_valid, z_valid, marker='.', markersize=10, color='b', linestyle='none')
            plt.plot(x_invld, z_invld, marker='.', markersize=10, color='r', linestyle='none')
            plt.plot(path_x, path_z, lw=2.5, color='orange', marker='x', markersize=15)

            x_rng, y_rng = self._world_size[0]/2, self._world_size[1]/2
            ax.set_xlim(-x_rng, x_rng)
            ax.set_ylim(-y_rng, y_rng)

            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        return shortest_path, path_dist.get(end)


# =========================================================================================== #

# ************** #
# Test functions #
# ************** #

# # Base world for map creation
# base_world_file = 'ObjectFinder_base.wbt'

# # Output file to use for new simulations
# test_world = 'ObjectFinder_TEST.wbt'

# # Base world for map creation
# base_world_file = 'ObjectFinder_base.wbt'

# # Output file to use for new simulations
# test_world = 'ObjectFinder_TEST.wbt'

# # Order of names should match order in Scene Tree
# obstacle_names = ['CardboardBox', 
#                   'PlasticCrate', 
#                   'WoodenBox', 
#                   'WoodenPalletStack', 
#                   'OilBarrel']

# # Robot size from documentation
# robotSize = 0.074  # 7.4cm --> m diameter
# OFFSET = robotSize/2  # for `validate_nodes` method

# # Directory for object images
# img_dir = '/home/tamer/UChiMSCA/MSCA32019_RealTimeSystems/Project/robotics/images'

# # Create world map
# myWorld = worldMap(base_world_file)
# myWorld.get_world_size()
# myWorld.get_obstacles()
# myWorld.make_grid()
# myWorld.validate_nodes(OFFSET)

# points_on_map = [(0.8022478826811765, 0.050964583933198915), 
#                  (-0.8750591591621408, 0.2177685991441365)]  # picked from plot

# starting_pt, endpoint = points_on_map[0], points_on_map[1]


# # Create node graph and test path finding
# myGraph = worldGraph(myWorld)
# myGraph.find_shortest_path(starting_pt, endpoint)
# myGraph.get_node_distances()

# # find path
# bestPath, bestPath_length = myGraph.find_shortest_path(start=starting_pt, end=endpoint, plot=True)
# bestPathSimple = simplify_path(bestPath)
# bestPath_angles, bestPath_dists = path_to_sequence(bestPathSimple)


