#!/usr/bin/env python

import numpy as np
import pybullet as p

from ravens.tasks import Task
from ravens import utils


class Insertion(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 3
        self.metric = 'pose'
        self.primitive = 'pick_place'

    def reset(self, env):
        self.num_steps = 1
        self.goal = {'places': {}, 'steps': []}

        # Add L-shaped block. (Should be ID=4)
        block_size = (0.1, 0.1, 0.04)
        block_urdf = 'assets/insertion/ell.urdf'
        block_pose = self.random_pose(env, block_size)
        block_id = env.add_object(block_urdf, block_pose)
        self.goal['steps'].append({block_id: (2 * np.pi, [0])})

        # Add L-shaped hole. (Should be ID=5)
        hole_urdf = 'assets/insertion/hole.urdf'
        hole_pose = self.random_pose(env, block_size)
        env.add_object(hole_urdf, hole_pose, fixed=True)
        self.goal['places'][0] = hole_pose


class InsertionTranslation(Insertion):

    def random_pose(self, env, object_size):
        position, rotation = super(InsertionTranslation, self).random_pose(env, object_size)
        rotation = p.getQuaternionFromEuler((0, 0, 0))
        return position, rotation
