#!/usr/bin/env python

import numpy as np
import pybullet as p

from ravens.tasks import Task
from ravens import utils as U


class InsertionGoal(Task):
    """Using insertion, but in a goal-based Transporters context."""

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 3
        self.metric = 'pose'
        self.primitive = 'pick_place'

    def reset(self, env, last_info=None):
        self.num_steps = 1
        self.goal = {'places': {}, 'steps': []}

        # Add L-shaped block.
        block_size = (0.1, 0.1, 0.04)
        block_urdf = 'assets/insertion/ell.urdf'
        block_pose = self.random_pose(env, block_size)
        block_id = env.add_object(block_urdf, block_pose)
        self.goal['steps'].append({block_id: (2 * np.pi, [0])})

        # Add L-shaped target pose, but without actually adding it.
        if self.goal_cond_testing:
            assert last_info is not None
            self.goal['places'][0] = self._get_goal_info(last_info)
            #print('\nin insertion reset, goal: {}'.format(self.goal['places'][0]))
        else:
            hole_pose = self.random_pose(env, block_size)
            self.goal['places'][0] = hole_pose
            #print('\nin insertion reset, goal: {}'.format(hole_pose))

    def _get_goal_info(self, last_info):
        """Used to determine the goal given the last `info` dict."""
        position, rotation, _ = last_info[4] # block ID=4
        return (position, rotation)
