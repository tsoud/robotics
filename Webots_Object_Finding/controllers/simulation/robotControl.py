"""
Tamer Abousoud

Main Robot Controls
---
Functions robot can perform:
- Move in a straight path
- Turn to a given angle
- Take pictures/video
- Return GPS coordinates
- Return sensor data (e.g. distance sensors, accelerometer, gyro)
"""

import os
import sys
import time
import math
import datetime as dt
import numpy as np
from collections import deque, defaultdict

# os.getcwd()

# *** IMPORTANT! ***
# Add path to webots controller libraries to use external control
sys.path.append("/usr/local/webots/lib/controller/python38")

# import robot and devices
from controller import Robot, Motor, Camera, GPS, Accelerometer, Gyro, DistanceSensor

# Directory for captured images:
capturedImgDir = '/home/tamer/MSCA/MSCA32019_RealTimeSystems/Project/robotics/robot_images'


# Simulation parameters
TIME_STEP = 64
MAX_SPEED = 6.28  # rad/s

# E-Puck properties for use with various functions
# properties to calculate rotation angle
axle_length = 52e-3 # 52mm as provided in docs
motor_steps_per_sec = 1000  # provided in docs
# There is conflicting info in docs about e-puck max linear velocity
# 0.115 m/s was found to work best for resolving angular motion
max_linear_velocity =  0.115 # 0.125 # 6.28 * 20.5e-3 # 0.25 # m/s 
# Angle traversed per motor step at max speed
max_angle_per_motorstep = (max_linear_velocity/motor_steps_per_sec) / (axle_length/2)


class deviceOffException(Exception):
    '''Raise an error if a device is disabled'''
    pass

class WriteError(Exception):
    '''Raise if trying to write a read-only attribute'''
    pass


# Create a class to  define the robot and its functions
class epuckControl(Robot):
    ''' 
    E-Puck v1 control to use in the simulated world
    '''

    def initialize(self, time_step=TIME_STEP, max_speed=MAX_SPEED, starting_angle=0,\
        history_size=25):
        '''
        time_step: simulation time step in milliseconds
        max_speed: max rotational speed of robot wheel motors (rad/s)
        history_size: Number of steps to keep in history

        NOTE: Currently, history is only recorded when the robot is in motion
        '''
        self.time_step = time_step
        self.max_speed = max_speed
        # Robot devices
        self.camera = self.getCamera('camera')
        self.gps = self.getGPS('gps')
        self.leftMotor = self.getMotor('left wheel motor')
        self.rightMotor = self.getMotor('right wheel motor')
        # Initialize motors
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)
        # Initial conditions
        self.stopped = True
        self.camera_on = False
        self.camera_freq = 1
        self.save_imgs = False
        self.img_format = 'png'
        self.save_img_every_nframes = 4
        self.gps_on = False
        self.gps_freq = 1
        # Maintain robot history
        self._current_angle = starting_angle  # track orientation wrt world (radians)
        self._history = deque([], maxlen=history_size)
        self.__fmt = '%Y-%m-%d %H:%M:%S.%f'  # timestamp format

    @property
    def robot_angle(self):
        '''Robot's angle wrt to its forward +ve axis'''
        return self._current_angle

    @robot_angle.setter
    def robot_angle(self, value):
        raise WriteError(f'`robot_angle()` is read-only. If you want to rotate the robot use `turn({value})`')


    @staticmethod
    def angular_to_linear(value, reverse=False):
        ''' 
        Convert angular velocity to linear and vice-versa. Default is angular to linear.
        Uses the e-puck wheel properties for conversion.
        ---
        value: angular (rad/s) or linear velocity (m/s) to convert
        reverse: convert from linear to angular if True
        '''
        wheel_radius = 20.5e-3  # 20.5mm per e-puck specs

        if reverse:
            return value / wheel_radius
        
        return value * wheel_radius

    
    def turn(self, angle, speed:float):
        '''
        Rotate the robot about the vertical (y) axis
        --- 
        angle: rotation angle in degrees
               +ve (ccw) is left, -ve (cw) is right
        sampling_period: time between measurements in ms.
                        Should be equal to or less than
                        the simulation time step.
                        If less, should be evenly divisible
                        (e.g. if time step is 32 ms sampling_period should
                        be 32, 16, 8, 4 ...)
        speed: motor speed as fraction of top speed
            float between [0, 1]
        '''
        if speed > 1 or speed < -1:
            raise ValueError('Speed must be float between [0, 1]')

        # Set motor speed
        motor_speed = speed * self.max_speed
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate angle increment per time step
        motor_steps = (self.time_step * 1e-3) * motor_steps_per_sec * speed
        delta_angle = motor_steps * max_angle_per_motorstep
        total_angle = 0  # angle swept since rotation start
        # After n_steps, turn more slowly for better accuracy
        n_steps = int(0.95 * (abs(angle_rad) / delta_angle))
        reduction = 0.05

        step = 0
        direction = 1 if angle_rad > 0 else -1

        while True:
            step_info = {}

            if self.step(self.time_step) == -1 or abs(total_angle) > abs(angle_rad):
                break

            else:
                step += 1
                reduce = 1 if step < n_steps else reduction

                step_info['time'] = dt.datetime.now().strftime(self.__fmt)
                # Set motor speeds
                leftSpeed = -motor_speed * direction * reduce
                rightSpeed = motor_speed * direction * reduce
                # Rotate robot
                self.leftMotor.setVelocity(leftSpeed)
                self.rightMotor.setVelocity(rightSpeed)
                # Keep track of angle swept
                total_angle += delta_angle * reduce
                self._current_angle += delta_angle*reduce*direction

                if self.gps_on and step % self.gps_freq == 0:
                    step_info['position'] = self.gps.getValues()
                    step_info['speed'] = self.gps.getSpeed()

                if (self.camera_on and self.save_imgs) and (step %\
                     self.camera_freq*self.save_img_every_nframes == 0):
                    self.save_camera_img(output=self.img_format)
                    step_info['image_data'] = self.camera.getImageArray()

                self._history.append(step_info)  # update history
        
    
    def move(self, speed, distance, reverse=False):
        '''
        speed: motor speed as fraction of top speed
               float between [0, 1]
        distance: distance to travel in meters
        '''
        if speed > 1 or speed < 0:
            raise ValueError('Speed must be float between [0, 1]')

        motor_speed = speed * self.max_speed
        # motor_speed is angular velocity; convert to linear increment
        dist_increment = self.angular_to_linear(motor_speed) * (self.time_step * 1e-3)
        dist_traveled = 0
        # For accuracy, slow down before reaching endpoint at n_steps
        n_steps = int(distance // dist_increment)
        # After n_steps reduce speed
        reduction = 0.2  # reduce to 1/5

        self.stopped = False
        rvrs = -1 if reverse else 1

        step = 0  # counter for frequency-dependent functions

        while True:
            step_info = {}  # collect info for history

            if self.step(self.time_step) == -1 or dist_traveled > distance:
                break

            else:
                step += 1
                reduce = 1 if step < n_steps else reduction

                # Adjust motor speeds to get to end
                self.leftMotor.setVelocity(motor_speed * reduce * rvrs)
                self.rightMotor.setVelocity(motor_speed * reduce * rvrs)
                # Update distance
                dist_traveled += dist_increment * reduce

                step_info['time'] = dt.datetime.now().strftime(self.__fmt)

                if self.gps_on and step % self.gps_freq == 0:
                    step_info['position'] = self.gps.getValues()
                    step_info['speed'] = self.gps.getSpeed()

                if (self.camera_on and self.save_imgs) and (step %\
                     self.camera_freq*self.save_img_every_nframes == 0):
                    self.save_camera_img(output='jpg')
                    step_info['image_data'] = self.camera.getImageArray()
                
                self._history.append(step_info)  # update history


    def stop(self):

        if not self.stopped:
            self.leftMotor.setVelocity(0.0)
            self.rightMotor.setVelocity(0.0)
            self.stopped = True


    def idle(self, wait_time):
        '''Idle the robot for `wait_time` seconds'''
        if not self.stopped:
            self.stop()
        time.sleep(wait_time)


    ### GPS FUNCTIONS ###

    def start_gps(self, frequency:int=1):
        '''
        frequency: sampling period relative to time step. 
                   e.g. `frequency` = 2 returns coordinates 
                   every 2 time steps. Should be int >= 1
        '''
        if frequency < 1:
            raise ValueError('`frequency` should be an int >= 1')

        if not isinstance(frequency, int):
            frequency = int(np.round(frequency))
            print(f'Non-integer given for `frequency`, value rounded to {frequency}')

        self.gps_freq = frequency

        if not self.gps_on:
            self.gps.enable(self.time_step * self.gps_freq)
            self.gps_on = True

    def stop_gps(self):

        if self.gps_on:
            self.gps.disable()
            self.gps_on = False

        
    ### CAMERA FUNCTIONS ###

    def start_camera(self, frequency:int=1, save_imgs=False):
        '''
        frequency: sampling period relative to time step. 
                   e.g. `frequency` = 2 returns image 
                   every 2 time steps. Should be int >= 1
        '''
        if frequency < 1:
            raise ValueError('`frequency` should be an int >= 1')

        if not isinstance(frequency, int):
            frequency = int(np.round(frequency))
            print(f'Non-integer given for `frequency`, value rounded to {frequency}')

        self.camera_freq = frequency

        self.camera.enable(self.time_step * frequency)
        self.camera_on = True
        self.save_imgs = save_imgs
    
    def stop_camera(self):
        self.camera.disable()
        self.camera_on = False

    def save_camera_img(self, img_dir=capturedImgDir, output:str='png', quality=90,\
        every_nframes:int=4):
        '''
        img_dir: Save images to this directory
        output: Either 'png' or 'jpg'
        quality: Only for jpg, from 1 (worst) to 100 best
        every_nframes: Save only the nth frame (e.g. 4 saves every 4th frame)
        '''
        img_num = len(os.listdir(img_dir)) + 1
        img_name = 'robot_img' + str(img_num).zfill(5) + '.' + output
        file_name = '/'.join([img_dir, img_name])

        self.save_img_every_nframes = every_nframes
        self.camera.saveImage(file_name, quality)


# ========================================================================================== #

# Some simple processes to test functions

# robot = epuckControl()

# robot.initialize()
# print(robot._history)
# robot.idle(2)
# robot.start_camera()
# robot.turn(90, 0.1)
# robot.move(0.4, 0.4)
# robot.idle(3)
# robot.start_gps(frequency=4)
# robot.gps_coords()
# robot.turn(-50, 0.05)
# robot.move(0.25, 1.0)
# robot.stop_camera()
# robot.stop_gps()
# robot.idle(2)
# # print(robot._history)
# robot.turn(30, 0.05)