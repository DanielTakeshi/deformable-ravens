#!/usr/bin/env python

import time
import numpy as np
import cv2
import pybullet as p
import matplotlib.pyplot as plt
from .task import Task
# from experimental.andyzeng.common import utils

from ravens import utils


class HanoiNew(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 14
        self.metric = 'pose'
        self.name = 'hanoinew'
        self.primitive = 'pick_place'
        
    def reset(self, env):

        # Random base pose
        rx = env.bounds[0, 0] + 0.2 + np.random.rand() * 0.1
        ry = env.bounds[1, 0] + 0.2 + np.random.rand() * 0.6
        rtheta = np.random.rand() * 2 * np.pi
        self.base_pos = np.array([rx, ry, 0.005])
        self.base_rot = p.getQuaternionFromEuler((0, 0, rtheta))
        base_urdf = 'assets/hanoi/stand.urdf'
        p.loadURDF(base_urdf, self.base_pos, self.base_rot, useFixedBase=1)

        # Rod poses in base coordinates
        self.rod_pos = [[0, -0.12, 0.03], [0, 0, 0.03], [0, 0.12, 0.03]]
        self.rod_pos = np.array(self.rod_pos)

        # Add disks
        env.obj = []
        if self.mode == 'train':
            self.n_disks = np.random.choice([2, 3, 5, 6])
        else:
            self.n_disks = np.random.choice([4, 7])
        for i in range(self.n_disks):
            urdf = 'assets/hanoi/slimdisk.urdf'
            pos = self.base2origin(self.rod_pos[0, :])
            pos[2] += 0.006 * (self.n_disks - i)
            obj = p.loadURDF(urdf, pos)
            cw = (self.n_disks - i - 1) / (self.n_disks - 1)
            color = [cw, 0.6, cw, 1.0]
            p.changeVisualShape(obj, -1, rgbaColor=color)
            env.obj.append(obj)

        self.goal = {'places': {}, 'steps': []}

        # Solve Hanoi sequence with dynamic programming
        steps = []  # [[obj id, from rod, to rod], ...]
        def solve_hanoi(n, t0, t1, t2):
            if n == 0:
                steps.append([n, t0, t1])
                return
            solve_hanoi(n - 1, t0, t2, t1)
            steps.append([n, t0, t1])
            solve_hanoi(n - 1, t2, t1, t0)
        solve_hanoi(self.n_disks - 1, 0, 2, 1)
        self.n_steps = len(steps)

        self.max_steps = min(self.n_steps + 10, self.n_steps * 2)

        # Construct goal sequence [{obj id : (symmetry, pose)}, ...]
        for step in steps:
            obj = env.obj[step[0]]
            pos = self.base2origin(self.rod_pos[step[2], :])
            rot = p.getQuaternionFromEuler((0, 0, 0))
            place_i = len(self.goal['places'])
            self.goal['places'][place_i] = (pos, rot)
            self.goal['steps'].append({obj: (0, [place_i])})

    # def origin2base(pos):

    def base2origin(self, pos):
        pos = pos.copy()
        theta = p.getEulerFromQuaternion(self.base_rot)[2]
        x = np.cos(theta) * pos[0] - np.sin(theta) * pos[1] + self.base_pos[0]
        y = np.sin(theta) * pos[0] + np.cos(theta) * pos[1] + self.base_pos[1]
        z = pos[2] + self.base_pos[2]
        return np.array([x, y, z])
