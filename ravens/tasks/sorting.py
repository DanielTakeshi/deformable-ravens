#!/usr/bin/env python

import numpy as np
import pybullet as p

from ravens.tasks import Task
from ravens import utils


class Sorting(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 10
        self.metric = 'pose'
        self.primitive = 'pick_place'
        self.position_eps = 0.045

    def reset(self, env):
        bowl_size = (0.12, 0.12, 0)
        block_size = (0.04, 0.04, 0.04)
        num_bowls = np.random.randint(1, 4)
        num_blocks = np.random.randint(1, num_bowls + 1)

        self.num_steps = num_blocks
        self.goal = {'places': {}, 'steps': [{}]}

        # Add bowls.
        bowl_urdf = 'assets/bowl/bowl.urdf'
        for i in range(num_bowls):
            bowl_pose = self.random_pose(env, bowl_size)
            env.add_object(bowl_urdf, bowl_pose, fixed=True)
            self.goal['places'][i] = bowl_pose

        # Add blocks.
        block_urdf = 'assets/stacking/block.urdf'
        possible_places = list(self.goal['places'].keys())
        for i in range(num_blocks):
            block_pose = self.random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            self.goal['steps'][0][block_id] = (0, possible_places)

        # Colors of distractor objects.
        bowl_colors = [utils.COLORS['purple'],
                       utils.COLORS['blue'],
                       utils.COLORS['yellow'],
                       utils.COLORS['orange'],
                       utils.COLORS['red']]
        block_colors = [utils.COLORS['purple'],
                        utils.COLORS['blue'],
                        utils.COLORS['yellow'],
                        utils.COLORS['orange'],
                        utils.COLORS['green']]

        # Add distractors.
        num_distractors = 0
        while num_distractors < 10:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = block_colors if is_block else bowl_colors
            pose = self.random_pose(env, size)
            if not pose:
                continue
            object_id = env.add_object(urdf, pose)
            color = colors[num_distractors % len(colors)]
            p.changeVisualShape(object_id, -1, rgbaColor=color + [1])
            num_distractors += 1
