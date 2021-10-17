'''
Tamer Abousoud

Map Simulated World
---
Use Webots .wbt file to create a map of the world and obstacles. Used to navigate
the robot through the world and locate items.

NOTE:
The robot used in this project is a GCtronic e-puck. Everything is configured
around this specific robot.

NOTE:
It is important for the world to be formatted properly in Webots for 
everything here to work properly.
'''

# standard libraries
import os
import math
import random
from pathlib import Path
from itertools import product
from collections import deque, defaultdict

# external libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.patches as patches


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

# Robot size from documentation
robotSize = 0.074  # 7.4cm --> m diameter
OFFSET = robotSize/2  # for `validate_nodes` method

# Directory for object images
img_dir = '/home/tamer/MSCA/MSCA32019_RealTimeSystems/Project/robotics/images'


# ----------------------------------------------
#          *** OBJECTS TO FIND ***
# Use a template based on the .wbt file to place
# a picture of an object in the world

def add_image_to_world(url:str, translation, rotation, name:str, size=(0.16, 0.12)):
    ''' 
    Uses Webots VRML formatting to add a picture of an object
    ---
    url: url or path for image to add
    translation: x, z location of image in world
    rotation: image rotation about Y-axis (degrees)
    name: a name for the image
    size: size of image
    '''
    template_str = '''DEF objectPic Solid {
  translation 0 0 0
  rotation 0 1 0 0
  children [
    Shape {
      appearance Appearance {
        texture ImageTexture {
          url [
            ""
          ]
        }
      }
      geometry Rectangle {
        size 0.16 0.12
      }
    }
  ]
  name ""
}'''

    tr_x, tr_y, tr_z = translation[0], size[1]/2, translation[1]
    rot = math.radians(rotation)

    template = dict(enumerate(template_str.split('\n')))

    template[1] = f'  translation {tr_x} {tr_y} {tr_z}'
    template[2] = f'  rotation 0 1 0 {rot}'
    template[8] = f'            "{url}"'
    template[13] = f'        size {size[0]} {size[1]}'
    template[17] = f'  name "{name}"'

    # return '\n'.join(template.values())
    return list(template.values())


#               ******************************************
#               *                                        *
#               *            *** ADD ROBOT ***           *
#               *                                        *
#               ******************************************

# Insert an e-puck robot into the .wbt file

def place_robot_in_world(translation, rotation, camera_size=(320, 320)):
    ''' 
    Uses Webots VRML formatting to insert e-puck robot into world
    ---
    translation: x, z location of robot in world
    rotation: robot rotation about Y-axis (degrees)
    camera_size: customize camera w x h
    '''
    template_str = '''E-puck {
  translation 0.0 0.0 0.0
  rotation 0.0 1.0 0.0 0.0
  controller "<extern>"
  camera_width 160
  camera_height 120
  turretSlot [
    GPS {
    }
  ]
}'''

    tr_x, tr_z = translation[0], translation[1]
    rot = math.radians(rotation)

    template = dict(enumerate(template_str.split('\n')))

    template[1] = f'  translation {tr_x} 0.0 {tr_z}'
    template[2] = f'  rotation 0 1 0 {rot}'
    template[4] = f'  camera_width {camera_size[0]}'
    template[5] = f'  camera_height {camera_size[1]}'

    # return '\n'.join(template.values())
    return list(template.values())


# =========================================================================================== #


class worldMap(object):
    '''
    Read the Webots world (.wbt) file to describe objects.
    Use this to instantiate a new world with a randomly-located robot,
    randomly place objects to find, and map a path.
    '''

    def __init__(self, world_file:str):
        ''' 
        world_file: filename of .wbt file
                    File should be located in the current project directory
        '''

        if not isinstance(world_file, str) and not world_file.endswith('wbt'):
            raise TypeError('File name for world should be string having .wbt extension')

        self.world_file = world_file

        curr_dir = Path.cwd().parent
        project_dir = curr_dir.parent

        if 'worlds' in os.listdir(project_dir):
            self.world_dir = project_dir.joinpath('worlds')
        else:
            raise FileNotFoundError('Could not find `worlds` folder in project directory')

        if self.world_file in os.listdir(self.world_dir):
            
            self.world_filepath = str(self.world_dir.joinpath(self.world_file))
        else:
            raise FileNotFoundError(f'No file named {self.world_file!r} in `worlds` folder')

        with open(self.world_filepath, 'r') as WF:
            world_info = WF.readlines()
        
        self._world_info_raw = world_info.copy()

        world_info = [line.strip() for line in world_info]
        self.world_info = ' '.join(world_info)

        # Initialize placeholders for map properties
        # Empty until appropriate function is called
        self.size = None
        self.grid = None
        self.obstacles = {}
        self.valid_nodes = []
        self._invalid_nodes = []
        self._obstacle_shapes = []

        # Options for creating new simulations
        self.object_locations = []
        self.starting_positions = []


    def get_world_size(self):
        '''
        Parse `world_info` to get planar (2D) dimensions of the world.
        Size is in meters.
        '''

        size_idx1 = self.world_info.find('floorSize ') + len('floorSize ')
        size_idx2 = self.world_info.find(' floorAppearance')

        size = self.world_info[size_idx1:size_idx2].split(' ')

        self.size = [float(s) for s in size]

    
    @staticmethod
    def get_obst_values(obstacle):
        '''Get 2D coordinates and  dimensions of obstacle'''

        coords_idx = obstacle.find('translation ') + len('translation ')
        coords = obstacle[coords_idx:].split(' ')[:3:2]  # only need X, Z values
        coords = [float(c) for c in coords]

        if obstacle.startswith('WoodenPalletStack'):
            dim = 'palletSize '
        elif obstacle.startswith('OilBarrel'):
            dim = 'radius '
        else:
            dim = 'size '

        dims_idx = obstacle.find(dim) + len(dim)
        dims = obstacle[dims_idx:].split(' ')
        dims = dims[:3:2] if len(dims) > 1 else dims
        dims = [float(d) for d in dims]

        if len(dims) > 1 and not math.isclose(dims[0], dims[1], rel_tol=0.1):
            # Flip dimensions if obstacle is not circular or nearly square
            # and has orientation close to 90 or its multiples
            if 'rotation' in obstacle:
                ortho = round(math.radians(90), 4)
                rot_idx = obstacle.find('rotation ') + len('rotation ')
                rotation = float(obstacle[rot_idx:].split(' ')[:4][-1])
                rotation = round(abs(rotation) % math.pi, 4)
                if math.isclose(rotation, ortho, rel_tol=0.1):
                    # dimensions are rotated
                    dims = [dims[1], dims[0]]

        return coords, dims


    def get_obstacles(self, obstacle_names:list=obstacle_names):
        ''' 
        Parse `world_info` to get locations and dimensions of obstacles
        '''

        obstacles = []

        obstacle_idx = []
        obstacle_idx_end = []

        for obst in obstacle_names:
            obst_start = [i[0] for i in enumerate(self._world_info_raw) 
                          if i[1].startswith(obst)]
            obstacle_idx.extend(obst_start)

        for idx in obstacle_idx:
            obstacle_idx_end.append(idx + self._world_info_raw[idx:].index('}\n'))

        for idx, idx_end in zip(obstacle_idx, obstacle_idx_end):
            obst = self._world_info_raw[idx:idx_end + 1]
            obst = [o.strip() for o in obst]
            obstacles.append(' '.join(obst))

        for obstacle in obstacles:
            obstacle_vals = {}
            values = self.get_obst_values(obstacle)
            obstacle_vals['coordinates'] = values[0]
            obstacle_vals['dimensions'] = values[1]
            obstacle_name = obstacle[obstacle.find('"')+1 : obstacle.rfind('"')]
            # Add to obstacles dict
            self.obstacles[obstacle_name] = obstacle_vals

        
    def make_grid(self, cell_size=robotSize):
        '''
        Create a grid of nodes corresponding to coordinates on the map.
        NOTE: (x, z) = (0,0) is located at the centroid of the world floor
              and the Y-axis projects normal to the robot's motion plane
        --- --- ---
        cell_size: size for dividing map plane into square cells.
                   Nodes are located at centroids of cells      
        '''
        if self.size is None:
            self.get_world_size()

        x_limits = round(self.size[0] - cell_size, 4)
        x_cells = int(round(x_limits/cell_size, 0))
        x_ticks = np.linspace(-x_limits/2, x_limits/2, x_cells)

        z_limits = round(self.size[1] - cell_size, 4)
        z_cells = int(round(z_limits/cell_size, 0))
        z_ticks = np.linspace(-z_limits/2, z_limits/2, z_cells)

        self.__nodeGrid = np.meshgrid(x_ticks, z_ticks)
        self._nodes = np.array(self.__nodeGrid).T

        self.grid = [tuple(self._nodes[i, j]) for i, j 
                     in product(range(x_cells), range(z_cells))]


    def validate_nodes(self, offset=0):
        ''' 
        Determine if nodes on the grid can be occupied by the robot.
        A node is invalid if it overlaps an obstacle or is so close 
        to an obstacle that the robot body would overlap the obstacle
        boundary.
        NOTE: Make sure to run `get_obstacles()` first, otherwise all
              nodes will be valid.
        --- --- ---
        offset: amount to offset obstacle boundaries so robot body doesn't 
                overlap.
        '''
        if self.grid is None:
            self.make_grid()

        valid_nodes = deque(self.grid)
        invalid_nodes = deque([])
        shapes = []  # obstacle shapes for validation
        real_shapes = [] # actual shapes for plotting

        # Codes for constructing polygon
        codes = [mplPath.Path.MOVETO, 
                 mplPath.Path.LINETO, 
                 mplPath.Path.LINETO, 
                 mplPath.Path.LINETO, 
                 mplPath.Path.CLOSEPOLY]

        # Calculate 2D boundary for each obstacle
        for obstacle in self.obstacles.keys():
            coords = self.obstacles.get(obstacle).get('coordinates')
            dims = self.obstacles.get(obstacle).get('dimensions')

            if len(dims) > 1:
                # If obstacle is square/rectangle use corners
                X_min = coords[0] - (dims[0]/2 + offset)
                Z_min = coords[1] - (dims[1]/2 + offset)
                X_max = coords[0] + (dims[0]/2 + offset)
                Z_max = coords[1] + (dims[1]/2 + offset)

                vertices = [(X_min, Z_min), 
                            (X_max, Z_min), 
                            (X_max, Z_max), 
                            (X_min, Z_max),
                            (X_min, Z_min)]
                
                shape = mplPath.Path(vertices, codes)
                
                if offset != 0:
                    # Actual shape for plot (no offset)
                    X_min_real = coords[0] - (dims[0]/2)
                    Z_min_real = coords[1] - (dims[1]/2)
                    X_max_real = coords[0] + (dims[0]/2)
                    Z_max_real = coords[1] + (dims[1]/2)

                    vertices_real = [(X_min_real, Z_min_real), 
                                     (X_max_real, Z_min_real), 
                                     (X_max_real, Z_max_real), 
                                     (X_min_real, Z_max_real),
                                     (X_min_real, Z_min_real)]

                    shape_real = mplPath.Path(vertices_real, codes)

            elif len(dims) == 1:
                # Use center and radius
                center = coords
                radius = dims[0] + offset

                shape = mplPath.Path.circle(center=center, radius=radius, readonly=True)

                if offset != 0:
                    # Actual shape for plot
                    radius_real = dims[0]
                    shape_real = mplPath.Path.circle(center=center, radius=radius_real,
                                                     readonly=True)

            shapes.append(shape)

            if 'shape_real' in locals():
                real_shapes.append(shape_real)
                
            for _ in range(len(valid_nodes)):
                node = valid_nodes.popleft()
                # Check if node is valid
                if shape.contains_point(node):
                    invalid_nodes.append(node)
                else:
                    valid_nodes.append(node)
            
        self.valid_nodes = list(valid_nodes)
        self._invalid_nodes = list(invalid_nodes)
        self._obstacle_shapes = real_shapes if len(real_shapes) > 0 else shapes.copy()


    def plot_map(self):
        ''' 
        Plot a 2D plane view of world showing obstacles and nodes
        ---
        BLUE nodes are VALID, RED nodes are INVALID
        '''
        x_lims = -np.ceil(self.size[0]/2), np.ceil(self.size[0]/2)
        y_lims = -np.ceil(self.size[1]/2), np.ceil(self.size[1]/2)

        x_valid, z_valid = zip(*self.valid_nodes)
        x_invld, z_invld = zip(*self._invalid_nodes)

        _, ax = plt.subplots()

        for shape in self._obstacle_shapes:
            patch = patches.PathPatch(shape, lw=1.0, facecolor='lime')
            ax.add_patch(patch)

        plt.plot(x_valid, z_valid, marker='.', color='b', linestyle='none')
        plt.plot(x_invld, z_invld, marker='.', color='r', linestyle='none')

        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)

        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    
    def define_node_neighbors(self):
        '''
        For each valid node, determine all immediate valid neighbors.
        Used by the path finding algorithm to determine shortest path.
        '''
        labeled_nodes = {idx:node if not node in self._invalid_nodes else 'invalid'\
            for idx, node in enumerate(self.grid)}

        neighbors = defaultdict(list)

        # Grid dimensions (n_rows x n_cols)
        n_rows, n_cols = self._nodes.shape[:2]

        # Find corner nodes
        upper_left = 0
        lower_left = n_rows - 1
        upper_right = n_rows*(n_cols - 1)
        lower_right = n_rows*n_cols - 1

        # Find edge nodes
        left = list(range(upper_left + 1, lower_left))
        right = list(range(upper_right + 1, lower_right))
        top = list(range(upper_left + n_rows, upper_right, n_rows))
        bottom = list(range(lower_left + n_rows, lower_right, n_rows))

        for idx, node in labeled_nodes.items():
            # Only look at valid nodes
            if node != 'invalid':
                neighbor_nodes = neighbors[node]
                # Corner nodes
                if idx == upper_left:
                    surrounding_ids = (idx+1, idx+n_rows, idx+n_rows+1)
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes                

                elif idx == lower_left:
                    surrounding_ids = (idx-1, idx+n_rows, idx+n_rows-1)
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes 

                elif idx == upper_right:
                    surrounding_ids = (idx+1, idx-n_rows, idx-n_rows+1)
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes 

                elif idx == lower_right:
                    surrounding_ids = (idx-1, idx-n_rows, idx-n_rows-1)
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes 
                
                # Edge nodes
                elif idx in left:
                    surrounding_ids = list([idx-1, idx+1] +\
                                        [idx+n_rows-1, idx+n_rows, idx+n_rows+1])
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes 
                
                elif idx in right:
                    surrounding_ids = list([idx-1, idx+1] +\
                                        [idx-n_rows-1, idx-n_rows, idx-n_rows+1])
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes 

                elif idx in top:
                    surrounding_ids = list([idx-n_rows, idx+n_rows] +\
                                        [idx+1-n_rows, idx+1, idx+1+n_rows])
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes 

                elif idx in bottom:
                    surrounding_ids = list([idx-n_rows, idx+n_rows] +\
                                        [idx-1-n_rows, idx-1, idx-1+n_rows])
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes 

                # All interior nodes
                else:
                    surrounding_ids = list([idx-1, idx+1] +\
                                        [idx-n_rows-1, idx-n_rows, idx-n_rows+1] +\
                                        [idx+n_rows-1, idx+n_rows, idx+n_rows+1])
                    neighbor_nodes = [labeled_nodes.get(i) for i in surrounding_ids
                                    if labeled_nodes.get(i) != 'invalid']
                    neighbors[node] = neighbor_nodes

        return neighbors

    
    def points_from_map(self, num_points):
        ''' 
        Select points from map interactively
        '''
        import matplotlib
        matplotlib.use('TkAgg')  # <-- Choose appropriate back-end
        import matplotlib.pyplot as plt

        self.plot_map()
        points = plt.ginput(num_points)

        return points


    def update_object_locations(self, num_locations):
        ''' 
        Select new points for object locations
        '''
        self.object_locations = self.points_from_map(num_locations)
        self.object_locations = np.round(self.object_locations, 5).T
        self.object_locations = list(zip(self.object_locations[0], 
                                         self.object_locations[1]))


    def update_robot_starting_pos(self, num_starting_pts):
        '''
        Select new robot starting points
        '''
        self.starting_positions = self.points_from_map(num_starting_pts)
        self.starting_positions = np.round(self.starting_positions, 5).T
        self.starting_positions = list(zip(self.starting_positions[0], 
                                           self.starting_positions[1]))


    def create_test_world(self, n_objects, object_locations=None, robot_position=None, 
                          image_directory=img_dir, output_file=test_world):
        ''' 
        Set up a new test world to run a simulation. Uses the map from the base
        world and adds the robot and objects to seek.

        Object locations and robot starting position are randomly selected 
        from `object_locations` and `starting_positions` attributes which can
        be given by `update_object_locations` and `update_robot_starting_pos`
        or entered manually.
        ---
        n_objects: number of object images to add
        object_locations: a list of coordinates from which to randomly choose 
                          object placements in map.
                          If `None` select from `object_locations` attribute.
        robot_position: coordinates of robot starting position.
                        If `None` select from `starting_positions` attribute.
        image_directory: directory of images for objects
        output_file: filename for new .wbt world
        '''

        # Create test world from base
        test_world = self._world_info_raw.copy()
        test_world = [l.rstrip('\n') for l in test_world]

        if object_locations is not None:
            assert len(object_locations) >= n_objects,\
                f'`n_objects` must be <= number of object locations'

            object_locations = random.sample(object_locations, k=n_objects)

        else:
            assert len(self.object_locations) >= n_objects,\
                f'`n_objects` must be <= number of object locations'

            object_locations = random.sample(self.object_locations, k=n_objects)

        if robot_position is not None:
            if abs(robot_position[0]) > self.size[0]/2 or\
                abs(robot_position[1]) > self.size[1]/2:
                raise ValueError(f'{robot_position} outside world boundaries')
            starting_position = robot_position

        else:
            starting_position = random.choice(self.starting_positions)


        img_dir = image_directory
        images = os.listdir(img_dir)
        test_imgs = random.sample(images, k=n_objects)

        # Add images
        for i in range(n_objects):
            url = str(Path(img_dir).joinpath(test_imgs[i]))
            translation = object_locations[i]
            rotation = round(random.uniform(0, 359), 1)
            name = test_imgs[i].split('.')[0]
            new_object = add_image_to_world(url=url, 
                                            translation=translation,
                                            rotation=rotation,
                                            name=name)
            test_world.extend(new_object)

        # Place new e-puck robot at start position
        # For simplicity, the robot does not currently use random rotation
        epuck_rotation = 0 #round(random.uniform(0, 359), 1)
        robot_start = place_robot_in_world(translation=starting_position, 
                                           rotation=epuck_rotation, 
                                           camera_size=(320, 320))
        
        test_world.extend(robot_start)
        # Output for .wbt file
        test_world_info = '\n'.join(test_world)
        # Save new test world in worlds directory
        test_world_file = str(self.world_dir.joinpath(output_file))
        with open(test_world_file, 'w') as file:
            file.write(test_world_info)


# ************************************************************************************** #

# *** Test ***

# myWorld = worldMap(base_world_file)
# myWorld.get_world_size()
# myWorld.get_obstacles()
# myWorld.make_grid()
# myWorld.validate_nodes(OFFSET)
# myWorld.plot_map()
# neighbors = myWorld.define_node_neighbors()


