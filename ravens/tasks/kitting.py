#!/usr/bin/env python

import os
import numpy as np

import pybullet as p

from ravens.tasks import Task
from ravens import utils


class Kitting(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 10
        self.metric = 'pose'
        self.primitive = 'pick_place'

    def reset(self, env):

        # Add kit.
        kit_size = (0.28, 0.2, 0.005)
        kit_urdf = 'assets/kitting/kit.urdf'
        kit_pose = self.random_pose(env, kit_size)
        env.add_object(kit_urdf, kit_pose, fixed=True)

        num_shapes = 20
        train_split = 14
        num_objects = 5
        if self.mode == 'train':
            object_shapes = np.random.randint(0, train_split, num_objects)
        else:
            object_shapes = np.random.randint(
                train_split, num_shapes, num_objects)

        self.num_steps = num_objects
        self.goal = {'places': {}, 'steps': [{}]}

        colors = [utils.COLORS['purple'],
                  utils.COLORS['blue'],
                  utils.COLORS['green'],
                  utils.COLORS['yellow'],
                  utils.COLORS['red']]

        symmetries = [2 * np.pi,
                      2 * np.pi,
                      2 * np.pi / 3,
                      np.pi / 2,
                      np.pi / 2,
                      2 * np.pi,
                      np.pi,
                      2 * np.pi / 5,
                      np.pi,
                      np.pi / 2,
                      2 * np.pi / 5,
                      0,
                      2 * np.pi,
                      2 * np.pi,
                      2 * np.pi,
                      2 * np.pi,
                      0,
                      2 * np.pi / 6,
                      2 * np.pi,
                      2 * np.pi]

        # Build kit.
        place_positions = [[-0.09, 0.045, 0.0014],
                           [0, 0.045, 0.0014],
                           [0.09, 0.045, 0.0014],
                           [-0.045, -0.045, 0.0014],
                           [0.045, -0.045, 0.0014]]
        object_template = 'assets/kitting/object-template.urdf'
        for i in range(num_objects):
            shape = f'{object_shapes[i]:02d}.obj'
            scale = [0.003, 0.003, 0.0001]  # .0005
            position = self.apply(kit_pose, place_positions[i])
            theta = np.random.rand() * 2 * np.pi
            rotation = p.getQuaternionFromEuler((0, 0, theta))
            replace = {
                'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
            urdf = self.fill_template(object_template, replace)
            box_id = env.add_object(urdf, (position, rotation), fixed=True)
            os.remove(urdf)
            self.goal['places'][i] = (position, rotation)

        # Add objects.
        for i in range(num_objects):
            ishape = object_shapes[i]
            size = (0.08, 0.08, 0.02)
            pose = self.random_pose(env, size)
            shape = f'{ishape:02d}.obj'
            scale = [0.003, 0.003, 0.001]  # .0005
            replace = {
                'FNAME': (shape,), 'SCALE': scale, 'COLOR': colors[i]}
            urdf = self.fill_template(object_template, replace)
            block_id = env.add_object(urdf, pose)
            os.remove(urdf)
            places = np.argwhere(ishape == object_shapes).reshape(-1)
            self.goal['steps'][0][block_id] = (
                symmetries[ishape], places)
