# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 20:38:47 2016

The Plant object should hold the state information for the limb system.  This
Allows generating velocity commands that are locally integrated to update position
as a function of time.  The update() method should be called repeatedly at fixed intervals
to advance the kinematic state

Usage:

from the python\minivie folder:

from Controls import Plant

Plant.main()

Revisions:
2016JUL26: Reverted changes back to the simple Joint dictionary since ROC table not working
2016OCT07: Added joint limit from xml file

@author: R. Armiger
"""

# Initial pass and simulating MiniVIE processing using python so that this runs on an embedded device
#
# Created 1/23/2016 Armiger
import os
import time
import logging
import numpy as np

import mpl.roc as roc
from mpl import JointEnum as MplId
from utilities import user_config


def class_map(class_name):
    """ Map a pattern recognition class name to a joint command

     The objective of this function is to decide how to interpret a class decision
     as a movement action.

     return JointId, Direction, IsGrasp, Grasp

       'No Movement' is not necessary in dict_Joint with '.get default return
     JointId, Direction = self.Joint.get(class_name,[ [], 0 ])
    """
    class_info = {'IsGrasp': None, 'JointId': None, 'Direction': 0, 'GraspId': None}

    # Map classes to joint id and direction of motion
    # Class Name: IsGrasp, JointId, Direction, GraspId
    class_lookup = {
        'No Movement': [False, None, 0, None],
        'Shoulder Flexion': [False, MplId.SHOULDER_FE, +1, None],
        'Shoulder Extension': [False, MplId.SHOULDER_FE, -1, None],
        'Shoulder Adduction': [False, MplId.SHOULDER_AB_AD, +1, None],
        'Shoulder Abduction': [False, MplId.SHOULDER_AB_AD, -1, None],
        'Humeral Internal Rotation': [False, MplId.HUMERAL_ROT, +1, None],
        'Humeral External Rotation': [False, MplId.HUMERAL_ROT, -1, None],
        'Elbow Flexion': [False, MplId.ELBOW, +1, None],
        'Elbow Extension': [False, MplId.ELBOW, -1, None],
        'Wrist Rotate In': [False, MplId.WRIST_ROT, +1, None],
        'Wrist Rotate Out': [False, MplId.WRIST_ROT, -1, None],
        'Wrist Adduction': [False, MplId.WRIST_AB_AD, +1, None],
        'Wrist Abduction': [False, MplId.WRIST_AB_AD, -1, None],
        'Wrist Flex In': [False, MplId.WRIST_FE, +1, None],
        'Wrist Extend Out': [False, MplId.WRIST_FE, -1, None],
        'Hand Open': [True, None, -1, None],
        # 'Spherical Grasp': [True, None, +1, 'Spherical Grasp'],
        # 'Tip Grasp': [True, None, +1, 'Tip Grasp'],
    }

    # rather than listing out all grasps, just list the arm motions and assume others are grasps

    if class_name in class_lookup:
        class_info['IsGrasp'], class_info['JointId'], class_info['Direction'], class_info['GraspId'] = class_lookup[
            class_name]
    else:
        # Assume this is a grasp in the ROC table
        # logging.warning('Unmatched class name {}'.format(class_name))
        class_info['IsGrasp'] = True
        class_info['JointId'] = None
        class_info['Direction'] = +1
        class_info['GraspId'] = class_name

    return class_info


class Plant(object):
    """
        The main state-space integrator for the system.
        Allows setting velocity commands and then as long as they execute
        the arm will move in that direction.  position parameters will be updated
        automatically so that position commands can be send to output devices
        Additionally limits are handled here as well as roc table lookup
    """

    def __init__(self, dt, roc_filename):

        # store current position and velocity commands
        self.joint_position = np.zeros(MplId.NUM_JOINTS)
        self.joint_velocity = np.zeros(MplId.NUM_JOINTS)

        # Load limits from xml config file
        self.lower_limit = np.zeros(MplId.NUM_JOINTS)
        self.upper_limit = np.zeros(MplId.NUM_JOINTS)

        # Basic time step for integration
        self.dt = dt

        # currently selected ROC for arm motion
        self.roc_id = ''
        self.roc_position = 0.0
        self.roc_velocity = 0.0

        # currently selected ROC for hand motion
        self.grasp_id = ''
        self.grasp_position = 0.0
        self.grasp_velocity = 0.0

        self.roc_filename = roc_filename
        self.roc_table = None

        self.load_roc()
        self.load_config_parameters()

    def load_config_parameters(self):
        # Load parameters from xml config file
        for i in range(MplId.NUM_JOINTS):
            limit = user_config.get_user_config_var(MplId(i).name + '_LIMITS', (0.0, 30.0))
            self.lower_limit[i] = np.deg2rad(limit[0])
            self.upper_limit[i] = np.deg2rad(limit[1])

    def load_roc(self):
        # load the roc table
        # can be run once plant is already initiated (reload)
        logging.info('Loading ROC table {}'.format(self.roc_filename))
        self.roc_table = roc.read_roc_table(self.roc_filename)

    def new_step(self):
        # set all velocities to 0 to prepare for a new time step
        # Typically followed by a call to setVelocity

        # reset velocity
        self.joint_velocity[:] = 0.0
        self.roc_velocity = 0.0
        self.grasp_velocity = 0.0

    def set_roc_velocity(self, roc_velocity):
        # set the velocities of the roc
        self.roc_velocity = roc_velocity

    def set_grasp_velocity(self, grasp_velocity):
        # set the velocities of the grasp roc
        self.grasp_velocity = grasp_velocity

    def set_joint_velocity(self, joint_id, joint_velocity):
        # set the velocities of the list of joint ids
        if joint_id is not None:
            self.joint_velocity[joint_id] = joint_velocity

    def update(self):
        # perform time integration based on elapsed time, dt

        # integrate roc values
        self.roc_position += self.roc_velocity * self.dt
        self.grasp_position += self.grasp_velocity * self.dt
        # Apply limits
        self.roc_position = np.clip(self.roc_position, 0, 1)
        self.grasp_position = np.clip(self.grasp_position, 0, 1)

        # integrate joint positions from velocity commands
        self.joint_position += self.joint_velocity * self.dt

        # set positions based on roc commands
        # hand positions will always be roc
        if self.roc_id in self.roc_table:
            new_vals = roc.get_roc_values(self.roc_table[self.roc_id], self.roc_position)
            self.joint_position[self.roc_table[self.roc_id].joints] = new_vals
        if self.grasp_id in self.roc_table:
            new_vals = roc.get_roc_values(self.roc_table[self.grasp_id], self.grasp_position)
            self.joint_position[self.roc_table[self.grasp_id].joints] = new_vals

        # Apply limits
        self.joint_position = np.clip(self.joint_position, self.lower_limit, self.upper_limit)


def main():
    # for demo
    from mpl.unity import UnityUdp as hSink

    # get default roc file.  This should be run from python\minivie as home, but 
    # also support calling from module directory (Utilities)
    roc_filename = "../../WrRocDefaults.xml"
    if os.path.split(os.getcwd())[1] == 'controls':
        roc_filename = '../' + roc_filename

    dt = 0.02
    p = Plant(dt, roc_filename)

    sink = hSink()
    sink.connect()

    print('LOWER LIMITS:')
    print(p.lower_limit)
    print('UPPER LIMITS:')
    print(p.upper_limit)

    time_begin = time.time()
    while time.time() < (time_begin + 1.5):  # main loop
        time.sleep(dt)
        class_name = 'Shoulder Flexion'
        class_info = class_map(class_name)

        # Set joint velocities
        p.new_step()
        # set the mapped class
        p.set_joint_velocity(class_info['JointId'], class_info['Direction'])
        # set a few other joints with a new velocity
        p.set_joint_velocity([MplId.ELBOW, MplId.WRIST_AB_AD], 2.5)

        # set a grasp state        
        # p.GraspId = 'rest'
        p.grasp_id = 'Spherical'
        p.set_grasp_velocity(0.5)

        p.update()

        sink.send_joint_angles(p.joint_position)

        # Print first 7 joints
        ang = ' '.join('{:6.1f}'.format(np.rad2deg(k)) for k in p.joint_position[:7])
        print('Angles (deg):' + ang + ' | Grasp Value: {:6.3f}'.format(p.grasp_position))

    class_map('Unmatched')

    sink.close()


# Main Function (for demo)
if __name__ == "__main__":
    main()
