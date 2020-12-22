#!/usr/bin/env python

import numpy as np
import pybullet as p


class RealSenseD415():
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Set default camera poses.
    front_position = (1., 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)

    # Default camera configs. (Daniel: setting Noise=True based on Andy advice)
    # Daniel: actually, getting some errors; let's revisit later.
    CONFIG = [
        {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': front_position,
            'rotation': front_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        },
        {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': left_position,
            'rotation': left_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        },
        {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': right_position,
            'rotation': right_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        }]
