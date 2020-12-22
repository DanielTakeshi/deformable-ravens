#!/usr/bin/env python

import numpy as np
import pybullet as p

from ravens.tasks import Task
from ravens import utils


class Sweeping(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'spatula'
        self.max_steps = 20
        self.metric = 'zone'
        self.primitive = 'sweep'

    def reset(self, env):
        self.total_rewards = 0

        # Add zone.
        zone_urdf = 'assets/zone/zone.urdf'
        self.zone_size = (0.12, 0.12, 0)
        self.zone_pose = self.random_pose(env, self.zone_size)
        env.add_object(zone_urdf, self.zone_pose, fixed=True)

        # Add morsels.
        self.object_points = {}
        morsel_urdf = 'assets/morsel/morsel.urdf'
        for i in range(50):
            rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
            ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
            position = (rx, ry, 0.01)
            theta = np.random.rand() * 2 * np.pi
            rotation = p.getQuaternionFromEuler((0, 0, theta))
            pose = (position, rotation)
            object_id = env.add_object(morsel_urdf, pose)
            # Why is this a fixed [-0.005, -0.005, -0.005] for each particle?
            self.object_points[object_id] = self.get_object_points(object_id)
