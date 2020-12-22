#!/usr/bin/env python
"""
The UR5 is centered at zone (0,0,0). The 'square' looks like this:

  |     |   xxxx
  |  o  |   xxxx
  |     |   xxxx
  -------

where `o` is the center of the robot, and `x`'s represent the workspace (so
the horizontal axis is x). Then the 'line' has to fill in the top part of the
square. Each edge has length `length` in code.
"""
import os
import time
import numpy as np
import pybullet as p

from ravens.tasks import Task
from ravens import utils


class Cable(Task):

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 20
        self.metric = 'zone'
        self.primitive = 'pick_place'

    def reset(self, env):
        self.total_rewards = 0
        self.goal = {'places': {}, 'steps': [{}]}

        # Hyperparameters for the cable and its `num_parts` beads.
        num_parts = 20
        radius = 0.005
        length = 2 * radius * num_parts * np.sqrt(2)

        # The square -- really 3 edges of it, since .urdf doesn't include a
        # 4th. The square_pose describes the pose relative to a coordinate
        # frame with (0,0,0) at the base of the UR5 robot. Replace the
        # dimension and lengths with desired values in a new .urdf.
        square_size = (length, length, 0)
        square_pose = self.random_pose(env, square_size)
        square_template = 'assets/square/square-template.urdf'
        replace = {'DIM': (length,), 'HALF': (length / 2 - 0.005,)}
        urdf = self.fill_template(square_template, replace)
        env.add_object(urdf, square_pose, fixed=True)
        os.remove(urdf)

        # Add goal line, to the missing square edge, enforced via the
        # application of square_pose on zone_position. The zone pose takes on
        # the square pose, to keep it rotationally consistent. The position
        # has y=length/2 because it needs to fill the top part of the square
        # (see diagram above in documentation), and x=0 because we later vary x
        # from -length/2 to length/2 when creating beads in the for loop. Use
        # zone_size for reward function computation, allowing a range of 0.03
        # in the y direction [each bead has diameter 0.01].
        line_template = 'assets/line/line-template.urdf'
        self.zone_size = (length, 0.03, 0.2)
        zone_range = (self.zone_size[0], self.zone_size[1], 0.001)
        zone_position = (0, length / 2, 0.001)
        zone_position = self.apply(square_pose, zone_position)
        self.zone_pose = (zone_position, square_pose[1])

        # Andy has this commented out. It's nice to see but not needed.
        #urdf = self.fill_template(line_template, {'DIM': (length,)})
        #env.add_object(urdf, self.zone_pose, fixed=True)
        #os.remove(urdf)

        # Add beaded cable.
        distance = length / num_parts
        position, _ = self.random_pose(env, zone_range)
        position = np.float32(position)
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius] * 3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 1.5)
        # part_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[radius] * 3)

        # Iterate and add to object_points, but interestingly, downstream code
        # only uses it for rewards, not for actions.
        self.object_points = {}
        for i in range(num_parts):
            position[2] += distance
            part_id = p.createMultiBody(
                0.1, part_shape, part_visual, basePosition=position)
            if len(env.objects) > 0:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=env.objects[-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, distance),
                    childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)
            if (i > 0) and (i < num_parts - 1):
                color = utils.COLORS['red'] + [1]
                p.changeVisualShape(part_id, -1, rgbaColor=color)
            env.objects.append(part_id)

            # To get target positions for each cable, we need initial reference
            # position `true_position`. Center at x=0 by subtracting length/2.
            # This produces a sequence of points like: {(-a,0,0), ..., (0,0,0),
            # ..., (a,0,0)}. Then apply zone_pose to re-assign `true_position`.
            # No need for orientation target values as beads are symmetric.
            self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)
            true_position = (radius + distance * i - length / 2, 0, 0)
            true_position = self.apply(self.zone_pose, true_position)
            self.goal['places'][part_id] = (true_position, (0, 0, 0, 1.))
            symmetry = 0  # zone-evaluation: symmetry does not matter
            self.goal['steps'][0][part_id] = (symmetry, [part_id])

        # Wait for beaded cable to settle.
        env.start()
        time.sleep(1)
        env.pause()
