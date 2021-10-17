# Object Finding with Path Planning in Webots

A tiny [E-Puck robot](https://www.gctronic.com/e-puck2.php) searches a map to locate and identify a certain object. This was part of a larger (incomplete) group project combining robotics, NLP and image recognition. At a high-level, the concept of the project was:
- Send an instruction (a text string) to the robot to find an object e.g. "Find the banana." 
- The robot parses and processes the command.
- The robot implements the path planning algorithms here to find the quickest way to get to each object (represented as a sign with a photorealistic image) on the map. This is the part of the project in this repo.
- When the robot reaches an object, it classifies the object in the image and returns the label as a caption.
- The search is over if the robot correctly finds and captions the correct object or fails to do so after going through all the objects.

The simulator used is [Webots](https://cyberbotics.com/) from Cyberbotics. It is a free, full-featured and open source robot simulator used in industry, research and education. The software has plenty of [documentation online](https://cyberbotics.com/doc/guide/index) and a [Discord Channel](https://discord.com/invite/nTWbN9m) where Webots developers actively engage with users.

<center><img src="./Screenshot from 2020-12-07 16-13-27.png" alt="E-Puck robot in the Webots simulator screen" width="1000"/></center>


<br>

## Directory and Files Information

To run simulations in Webots, it is necessary to set up a project folder with the proper directory structure. See the "Running this Simulation in Webots" and "Project Details - Robotics" files for more information.
The directory structure is set up automatically by Webots when running the "New Project Directory..." wizard. This project directory was set up by the wizard and follows Webots standard structure.

*NOTE*: When using the Webots simulator for this project, be sure this directory is set up and your current working directory when using the <extern> controller setting is `./robotics/controllers/simulation`

### Directory Folders:
---
**worlds**: 
Holds the .wbt world files for the simulations. For this project, this folder should contain two files. 'ObjectFinder_base.wbt' is an empty world without the objects and robot (just the maze) and 'ObjectFinder_TEST.wbt' is the same world after adding the robot and objects.

**protos**: 
This folder is empty for this project but it is used to hold any custom .proto files. These files are VRML97 format and describe entities in the worls like robots, shapes, physics, etc. No custom protos were made for this project.

**controllers**: 
This is where the bulk of the work was done for this project. Controllers can contain any number of scripts for controlling robots. All controllers are in a subfolder named "simulation". The main controller is `searchWorld.py` which imports the other files. There is a description of the general purpose in the heading of each script and more details in the functions but following a very brief summary:

`robotControl.py` - contains the actual robot controls (move this, measure that, etc.).<br>
`worldMap.py` - translates the world into a map of nodes and coordinates the robot can use to navigate and shows where the robot can or can't go.<br>
`findPath.py` - finds paths through the world using the maps, uses Dijkstra's shortest path algorithm to find shortest distance between points.<br>
`searchWorld.py` - searches the world for objects.<br>

**textures**:
Textures imported by Webots from its home directory to the project directory (gives items in the world their appearance).

**images**:
User created directory for storing test images for image detection.

**robot_images**:
User created directory for storing images taken by robot.

**Other folders**:
libraries, plugins were automatically created and are used for optional libraries and plugins. Not used for this project.
