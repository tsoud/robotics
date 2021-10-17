'''
Run the robot simulation.
---
While this script is running, the robot searches for items
and carries out various operations on them.
'''

# standard libraries
import os
import sys
# import subprocess
import time
import random
import math
import multiprocessing as mp
from multiprocessing import Process, Pool, Queue
import heapq as hp
from bisect import bisect
from itertools import product
from collections import defaultdict, deque, namedtuple
from pathlib import Path

# external libraries
import numpy as np
from scipy.spatial import distance
import matplotlib
# matplotlib.use('tkAgg')  # --> choose suitable backend for interactive plots
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.patches as patches

# os.getcwd()

# *** IMPORTANT! ***
# Add path to webots controller libraries to use external control
sys.path.append("/usr/local/webots/lib/controller/python38")

# import robot and devices
from controller import Robot, Motor, Camera, GPS, Accelerometer, Gyro, DistanceSensor

# import simulation scripts
from robotControl import *
from worldMap import *
from findPath import *

# %matplotlib tk
# %matplotlib inline

# ========================================================================================== #

# *** Some basic variables and parameters for the simulation ***

# Simulation parameters
TIME_STEP = 32 #64
MAX_SPEED = 6.28  # rad/s

# *** Create a new simulation world from the base world ***
# Base world for map creation
base_world_file = 'ObjectFinder_base.wbt'
# Output file to use for new simulations
test_world = 'ObjectFinder_TEST.wbt'
# Order of names should match order in Scene Tree
obstacle_names = ['CardboardBox', 
                  'PlasticCrate', 
                  'WoodenBox', 
                  'WoodenPalletStack', 
                  'OilBarrel']
# Path to world directory
world_dir = '/home/tamer/MSCA/MSCA32019_RealTimeSystems/Project/robotics/worlds'
# Robot size from documentation
robotSize = 0.074  # 7.4cm --> m diameter
OFFSET = robotSize/2  # for `validate_nodes` method

# Directory for object test images
img_dir = '/home/tamer/MSCA/MSCA32019_RealTimeSystems/Project/robotics/images'


# ========================================================================================== #

# *** Initialize the world map, graph for searching, and e-puck robot ***

# Define and create a new simulation world
myWorld = worldMap(base_world_file)
myWorld.get_world_size()
myWorld.get_obstacles()
myWorld.make_grid()
myWorld.validate_nodes(OFFSET)


# Uncomment between these lines to create a new world
# with new robot and object locations.
# Leave commented to use the last exisiting world.
# -----------------------------------------------------

# These functions allow picking points from a plot
# `create_test_world` will randomly select from these points 
# to place the robot and objects in the world

# myWorld.update_object_locations(num_locations=10)
# myWorld.update_robot_starting_pos(num_starting_pts=5)

# Hard-code points to avoid hassle of using matplotlib's 
# cumbersome backend management
# use_obj_pts = [(0.61138, -0.3057),
#                 (0.75676, 0.06684),
#                 (0.65552, 0.87293),
#                 (0.0662, 0.90668),
#                 (-0.48547, 0.84308),
#                 (-0.86061, 0.20833),
#                 (-0.77753, 0.55102),
#                 (-0.34009, -0.73016),
#                 (-0.78922, -0.77429),
#                 (0.80609, -0.8366)]

# use_start_pts = [(0.32711, 0.36799),
#                 (0.14928, 0.19535),
#                 (0.05841, 0.32775),
#                 (-0.09086, 0.19665),
#                 (-0.3349, -0.06556)]


# myWorld.object_locations = use_obj_pts
# myWorld.starting_positions = use_start_pts

# myWorld.create_test_world(n_objects=4)

# -----------------------------------------------------

# Create node graph for path finding
myGraph = worldGraph(myWorld)
myGraph.get_node_distances()

# myGraph.find_shortest_path(start=(-0.335, -0.0656), end=(0.757, 0.0668), plot=True)

# ========================================================================================== #

# *** Helper functions for searching ***

def get_object_locations(worldfile=test_world, world_dir=world_dir):
    '''
    Return locations of objects in the world.
    DO NOT return names of actual objects - That's 
    for the robot to figure out!
    ---
    worldfile, world_dir: filename and directory of .wbt file
    '''
    if not worldfile in os.listdir(world_dir):
        raise FileNotFoundError(f'{worldfile!r} not found in {world_dir}')

    locations = []
    file_wbt = '/'.join([world_dir, test_world])

    with open(file_wbt, 'r') as wf:
        world_info = wf.readlines()

    for i in range(len(world_info)):
        if 'objectPic' in world_info[i-1]:
            item = world_info[i].strip()  # translation line
            item = item.split(' ')
            item_coords = (float(item[1]), float(item[3]))  # only need x, z
            locations.append(item_coords)
    
    # return coordinates of items and number of items
    return len(locations), locations


def get_robot_starting_position(worldfile=test_world, world_dir=world_dir):
    '''
    Return robot location in world
    ---
    worldfile, world_dir: filename and directory of .wbt file
    '''
    if not worldfile in os.listdir(world_dir):
        raise FileNotFoundError(f'{test_world!r} not found in {world_dir}')

    file_wbt = '/'.join([world_dir, worldfile])

    with open(file_wbt, 'r') as wf:
        world_info = wf.readlines()

    for i in range(len(world_info)):
        if 'E-puck' in world_info[i-1]:
            position = world_info[i].strip()  # translation line
            position = position.split(' ')
            position = (float(position[1]), float(position[3]))  # only need x, z
    
    return position
    

def get_last_robot_position(epuckControl, worldfile=test_world, world_dir=world_dir):
    '''
    Returns the (x,z) coordinates of the robot's last position 
    from history
    ---
    worldfile, world_dir: filename and directory of .wbt file for initial location
    '''
    if len(epuckControl._history) > 1:
        position = epuckControl._history[-1].get('position')
        return (position[0], position[2])
    return get_robot_starting_position(worldfile, world_dir)


# ========================================================================================== #

# *** Main function for locating objects in the world ***

# Finds shortest paths, keeps track of what was found, etc.
      

class searchWorld:
    '''
    Searches world for objects.
    --- --- ---
    '''
    def __init__(self, epuckControl, worldGraph, testworld=test_world):
        ''' 
        epuckControl: control script to use
        worldGraph: `worldGraph` object of current world
        testworld: .wbt file of current world
        '''
        self.world = test_world
        self.graph = worldGraph
        self.start = get_robot_starting_position(self.world)
        self.last_position = self.start
        self.last_path = []  # path followed to last object
        self.n_objs, self.obj_locations = get_object_locations(self.world)
        # Use with multiprocessing for solving multiple paths concurrently
        self.__path_options = Queue()  # paths to solve
        self._current_paths = Queue()  # solved paths
        self._lock = mp.Lock()
        # List of solved paths to search
        self._paths = []


    def path_points(self):
        ''' 
        Get start and end points for path planning based on robot and 
        object locations.
        '''
        for endpoint in self.obj_locations:
            self.__path_options.put((self.last_position, endpoint))


    def get_path_to_obj(self):
        ''' 
        Calculate best path for an object in the world
        '''

        while not self.__path_options.empty():

            try:
                start, end = self.__path_options.get(block=False)
            except:
                pass

            else:
                with self._lock:
                    path, path_length = self.graph.find_shortest_path(start, end)
                    self._current_paths.put((path_length, path))

    
    def process_paths(self):
        ''' 
        Solve all object paths concurrently. Update list of path options.
        '''

        n_workers = self.n_objs
        processes = [Process(target=self.get_path_to_obj) for _ in range(n_workers)]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        self._paths = [self._current_paths.get() for p in processes]
    
    
    def select_closest_object(self):
        '''
        Select object with the shortest overall path.
        '''

        if len(self._paths) == 0:
            return 'No objects left in world'

        shortest_path = min(self._paths)
        self._paths.remove(shortest_path)
        self.n_objs = len(self._paths)
        self.obj_locations = [p[1][-1] for p in self._paths]
        # re-initialize the queues
        self.__path_options = Queue()
        self._current_paths = Queue()

        return shortest_path[1]
        

# ========================================================================================== #

# *** Moving and searching in the world ***


from findPath import simplify_path, path_to_sequence


def flip_coords(coords, direction='ud'):
    ''' 
    Flip path coordinates to realign with world
    ---
    coords: list of coordinates
    direction: 'ud' up/down (z-axis in Webots world)
               'lr' left/right (x-axis)
               'both'
    '''
    directions = {'ud': (1, -1), 'lr': (-1, 1), 'both': (-1, -1)}
    flip = directions.get(direction)
    flipped_coords = np.asarray(coords) * flip
    return list(zip(*flipped_coords.T))


def move_to_start(epuckControl, path, instr_seq, turn_speed=0.1,\
     move_speed=0.25, worldfile=test_world):
    ''' 
    Move robot to starting node of a new path and orient it so its forward
    movement aligns with the path.
    ---
    path: a simplified path (run `simplify_path()` first)
    instr_seq: instruction sequence from `path_to_sequence(path)`
    turn_speed: motor speed for turning
    move_speed: motor speed when moving straight
    '''
    # print("Robot angle at start:", epuckControl.robot_angle)
    # Use last position or start
    curr_position = get_last_robot_position(epuckControl)
    if curr_position is None:
        curr_position = get_robot_starting_position(worldfile)
    
    path_start = path[0]

    dist = distance.euclidean(path_start, curr_position)    
    x1, z1 = curr_position[0], curr_position[1]
    x2, z2 = path_start[0], path_start[1]
    dx, dz = x2 - x1, z2 - z1
    # 1st angle to turn robot towards path start
    turn_angle1 = math.degrees(math.acos(abs(dz)/dist))
    if dx > 0:  # determine if pt is left/right of robot
        turn_angle1 = -turn_angle1
    if dz > 0:  # determine if pt is in front/behind robot
        turn_angle1 = 180 - turn_angle1
    # Adjust for current robot pose
    turn_angle1 -= math.degrees(epuckControl.robot_angle)

    # Turn and move to starting position (slowly)
    # print("\nTurn angle 1:", turn_angle1)
    epuckControl.turn(angle=turn_angle1, speed=turn_speed)
    # print("\nDistance to start:", dist)
    epuckControl.move(speed=move_speed, distance=dist)
    # print("Robot angle:", math.degrees(epuckControl.robot_angle))


    # After moving to start, align robot so it faces 1st path segment
    start_angle = instr_seq.get('starting_angle')  # angle of first leg
    turn_angle2 = start_angle - math.degrees(epuckControl.robot_angle)
    # apply adjustment
    if abs(turn_angle2) > 180:  # quicker to go the other way
        sign = np.sign(turn_angle2)
        turn_angle2 = (360 - abs(turn_angle2)) * sign * (-1)
    # execute 2nd turn
    print("\nTurn angle 2:", turn_angle2, "\n")
    epuckControl.turn(angle=turn_angle2, speed=turn_speed)
    print("\nRobot angle:", math.degrees(epuckControl.robot_angle))


def follow_path(epuckControl, searchWorld, path, speed=0.5, idle_at_start=1):
    ''' 
    Convert path to a sequence of instructions and move along the path.
    ---
    path: the path to follow in points (simplified or unsilmplified)
    idle_at_start: wait for this many seconds after moving to the start of path
                   before moving along the path.
    flip_cs: Flip coordinates 'ud', 'lr', 'both' or None (don't flip)
    '''
    success = False  # True if end of path reached successfully

    path = simplify_path(path)

    # if flip_cs is not None:
    #     path = flip_coords(path, direction=flip_cs)

    instr_seq = path_to_sequence(path)

    # Move to starting point and align
    move_to_start(epuckControl, 
                  path=path, 
                  instr_seq=instr_seq)

    # print("Position:", epuckControl._history[-1].get('position'))   
    epuckControl.idle(idle_at_start)
    
    for turn, move in instr_seq.get('instructions'):
        # print(f'Turn: {turn}, Move: {move}')
        epuckControl.turn(angle=turn, speed=speed*0.2)
        epuckControl.move(distance=move, speed=speed)
        # print("Robot angle:", math.degrees(epuckControl.robot_angle))
        # print("Position:", epuckControl._history[-1].get('position'))

        # Placeholder for later...
        # if 'something happens to prevent robot from getting to end':
        #     do something or break and return success == False
        # This would also be a good place to add dynamic correction 
        # e.g. from GPS or proximity sensors

    # print('\n')
    success = True
    # sequence = zip(turns, moves)
    searchWorld.last_position = get_last_robot_position(epuckControl)

    return success  # useful for retracing steps


# ========================================================================================== #


# *** Start the actual simulation ***

# Launch Webots and initialize robot
# os.system('/usr/local/bin/webots')  # --> better to start manually

########################################################
#  NOTE: Start Webots before running next two lines!!  #
########################################################

myRobot = epuckControl()
myRobot.initialize(time_step=32)


# ========================================================================================== #

# Test search functions
# import pprint

# Initialize the object search
mySearch = searchWorld(myRobot, myGraph, test_world)
# Get paths to objects
# mySearch.path_points()
# mySearch.process_paths()


# mySearch._current_paths
# mySearch._paths
# mySearch.paths_to_objects()
# pprint.pprint(mySearch._paths)
# Select the closest item
# testPath1 = mySearch.select_closest_object()
# # print('\n')
# # pprint.pprint(testPath)
# mySearch.path_points()
# mySearch.process_paths()
# testPath2 = mySearch.select_closest_object()

# print(simplify_path(testPath1))
# print('\n')
# print(simplify_path(testPath2))






# # For debugging, hardcode shortest path:
# testPath = [(-0.34668, -0.03852),
#             (-0.42372, -0.11556),
#             (-0.50076, -0.1926),
#             (-0.5778, -0.26964),
#             (-0.65484, -0.34668),
#             (-0.73188, -0.42372),
#             (-0.73188, -0.50076),
#             (-0.73188, -0.5778),
#             (-0.73188, -0.65484),
#             (-0.73188, -0.73188),
#             (-0.80892, -0.80892)]


# testPath = [(-0.34668, -0.03852),
#             (-0.26964, -0.03852),
#             (-0.1926, -0.03852),
#             (-0.11556, -0.03852),
#             (-0.03852, 0.03852),
#             (0.03852, 0.03852),
#             (0.11556, 0.03852),
#             (0.1926, 0.11556),
#             (0.26964, 0.1926),
#             (0.34668, 0.26964),
#             (0.42372, 0.34668),
#             (0.50076, 0.42372),
#             (0.5778, 0.42372),
#             (0.65484, 0.42372),
#             (0.73188, 0.42372),
#             (0.80892, 0.42372),
#             (0.88596, 0.34668),
#             (0.88596, 0.26964),
#             (0.80892, 0.1926),
#             (0.80892, 0.11556),
#             (0.73188, 0.03852)]


# testPath = [(-0.34668, -0.03852),
#             (-0.34668, 0.03852),
#             (-0.34668, 0.11556),
#             (-0.34668, 0.1926),
#             (-0.34668, 0.26964),
#             (-0.26964, 0.34668),
#             (-0.1926, 0.42372),
#             (-0.11556, 0.50076),
#             (-0.03852, 0.5778),
#             (-0.03852, 0.65484),
#             (-0.03852, 0.73188),
#             (-0.03852, 0.80892),
#             (0.03852, 0.88596)]


# testPath = [(-0.34668, -0.03852),
#             (-0.26964, -0.11556),
#             (-0.1926, -0.1926),
#             (-0.11556, -0.26964),
#             (-0.03852, -0.34668),
#             (0.03852, -0.42372),
#             (0.11556, -0.50076),
#             (0.1926, -0.5778),
#             (0.26964, -0.65484),
#             (0.34668, -0.65484),
#             (0.42372, -0.73188),
#             (0.50076, -0.80892),
#             (0.5778, -0.80892),
#             (0.65484, -0.80892),
#             (0.73188, -0.80892),
#             (0.80892, -0.73188),
#             (0.80892, -0.65484),
#             (0.88596, -0.5778),
#             (0.88596, -0.50076),
#             (0.80892, -0.42372),
#             (0.73188, -0.34668),
#             (0.65484, -0.34668),
#             (0.5778, -0.26964)]



# ========================================================================================== #

# TEST

# def get_dist_trvld(start, end):
#     start = (start[0], start[2])
#     end = (end[0], end[2])
#     return distance.euclidean(start, end)

# myRobot.idle(1)
# myRobot.move(speed=0.2, distance=0.1)
# myRobot.turn(45, speed=0.05)
# myRobot.move(speed=0.2, distance=0.1)
# myRobot.turn(45, speed=0.05, clockwise=False)
# myRobot.move(speed=0.2, distance=0.1)

# print('\n')
# [print(p) for p in simplify_path(testPath)]

myRobot.idle(2)
myRobot.start_camera(2)

# # move_to_start(myRobot, testPath)
# A = get_robot_starting_position()
# # print(A)
myRobot.idle(2)
myRobot.start_gps()
# print(testPath)
# # print(myRobot.robot_angle)
# myRobot.move(speed=0.25, distance=0.1)
# A = (A[0], 0, A[1])
# B = myRobot._history[-1].get('position')
# print(B)
# # print(get_dist_trvld(A, B))
# # print(math.degrees(myRobot.robot_angle))
# myRobot.turn(angle=90, speed=0.05)
# myRobot.move(speed=0.25, distance=0.15)
# C = myRobot._history[-1].get('position')
# print(C)
# # print(get_dist_trvld(B, C))
# # print(math.degrees(myRobot.robot_angle))
# myRobot.turn(angle=-45, speed=0.05)
# myRobot.move(speed=0.25, distance=0.2)
# D = myRobot._history[-1].get('position')
# print(D)
# # print(get_dist_trvld(C, D))
# # print(math.degrees(myRobot.robot_angle))

mySearch.path_points()
mySearch.process_paths()
testPath1 = mySearch.select_closest_object()

follow_path(epuckControl=myRobot, searchWorld=mySearch, path=testPath1, speed=0.4)

myRobot.idle(2)

mySearch.path_points()
mySearch.process_paths()
testPath2 = mySearch.select_closest_object()

follow_path(epuckControl=myRobot, searchWorld=mySearch, path=testPath2, speed=0.4)

myRobot.idle(2)

mySearch.path_points()
mySearch.process_paths()
testPath3 = mySearch.select_closest_object()

follow_path(epuckControl=myRobot, searchWorld=mySearch, path=testPath3, speed=0.4)

myRobot.idle(2)

mySearch.path_points()
mySearch.process_paths()
testPath4 = mySearch.select_closest_object()

follow_path(epuckControl=myRobot, searchWorld=mySearch, path=testPath4, speed=0.4)


# myRobot.turn(72, speed=0.2)
# print(math.degrees(myRobot.robot_angle))
# myRobot.idle(2)
# myRobot.turn(-138.5, speed=0.2)
# print(math.degrees(myRobot.robot_angle))
# # myRobot.idle(2)
# # myRobot.turn(-18, speed=0.2)
# # print(myRobot.robot_angle)
# myRobot.idle(2)
# myRobot.turn(87.3, speed=0.2)
# print(math.degrees(myRobot.robot_angle))
# myRobot.idle(2)
# myRobot.turn(-26.7, speed=0.2)
# print(math.degrees(myRobot.robot_angle))
# # myRobot.idle(2)
# # myRobot.turn(51.1, speed=0.2)
# # print(myRobot.robot_angle)
# a = myRobot._history[-1].get('position')
# myRobot.idle(2)
# myRobot.move(distance=0.1, speed=0.2)
# b = myRobot._history[-1].get('position')
# angle = math.atan(abs(b[0]-a[0])/abs(b[2]-a[2]))
# print(math.degrees(angle))


# print(math.degrees(myRobot.robot_angle))
# print('\n')
# hist = myRobot._history
# print([h.get('position') for h in hist])

# ========================================================================================== #

# *** INCOMPLETE CODE FOR TAKING COMMANDS AND FINDING OBJECTS IN THE WORLD ***

# def search_for_object(object_name, epuckControl, worldMap, worldGraph, searchWorld):

#     object_found = False
#     item = object_name

#     while True:

#         if object_found or epuckControl.step(epuckControl.time_step) == -1:
#             break

#         else:
#             # Find paths and go to nearest object
#             searchWorld.paths_to_objects()
#             best_path = searchWorld.select_closest_object()

#             move_to_start(epuckControl, best_path)
#             follow_path(myRobot, searchWorld, testPath)

#             # At end of path, check object
#             epuckControl.idle(3)
#             # Send info to server
#             found_item = epuckControl._history[-1].get('image')
#             # Send to server ...
#             # Wait for response ...
#             if found_item == item:
#                 object_found = True


