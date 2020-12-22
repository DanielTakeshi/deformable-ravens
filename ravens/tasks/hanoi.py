#!/usr/bin/env python

import pybullet as p

from ravens.tasks import Task
from ravens import utils


class Hanoi(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 14
        self.metric = 'pose'
        self.primitive = 'pick_place'

    def reset(self, env):
        # Add stand.
        base_size = (0.12, 0.36, 0.01)
        base_urdf = 'assets/hanoi/stand.urdf'
        base_pose = self.random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, fixed=True)

        # Rod positions in base coordinates.
        rod_positions = ((0, -0.12, 0.03), (0, 0, 0.03), (0, 0.12, 0.03))

        # Add disks.
        num_disks = 3
        for i in range(num_disks):
            disk_urdf = 'assets/hanoi/disk%d.urdf' % i
            position = self.apply(base_pose, rod_positions[0])
            z_offset = 0.015 * (num_disks - i - 2)
            position = (position[0], position[1], position[2] + z_offset)
            env.add_object(disk_urdf, (position, base_pose[1]))

        # Solve Hanoi sequence with dynamic programming.
        hanoi_steps = []  # [[object index, from rod, to rod], ...]

        def solve_hanoi(n, t0, t1, t2):
            if n == 0:
                hanoi_steps.append([n, t0, t1])
                return
            solve_hanoi(n - 1, t0, t2, t1)
            hanoi_steps.append([n, t0, t1])
            solve_hanoi(n - 1, t2, t1, t0)
        solve_hanoi(num_disks - 1, 0, 2, 1)
        self.num_steps = len(hanoi_steps)

        # Construct goal sequence [{object id : (symmetry, pose)}, ...]
        self.goal = {'places': {}, 'steps': []}
        for step in hanoi_steps:
            object_id = env.objects[step[0]]
            rod_position = rod_positions[step[2]]
            place_position = self.apply(base_pose, rod_position)
            place_pose = (place_position, p.getQuaternionFromEuler((0, 0, 0)))
            place_id = len(self.goal['places'])
            self.goal['places'][place_id] = place_pose
            self.goal['steps'].append({object_id: (0, [place_id])})

        #print('\nJust finished reset()')
        #print('places: {}'.format(self.goal['places']))
        #print('steps: {}'.format(self.goal['steps']))
