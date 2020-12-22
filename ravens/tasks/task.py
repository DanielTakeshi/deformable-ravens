#!/usr/bin/env python
import sys
import os
import time
import string
import random
import collections

import cv2
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import ConvexHull

from ravens import tasks
from ravens import utils
from ravens import cameras

# An identity pose we can use to gracefully handle failure cases.
IDENTITY = {'pose0': ((0.3,0,0.3), (0,0,0,1)), 'pose1': ((0.3,0,0.3), (0,0,0,1))}
BEAD_THRESH = 0.33


class Task():

    def __init__(self):
        self.mode = 'train'

        # Evaluation epsilons (for pose evaluation metric).
        self.position_eps = 0.01
        self.rotation_eps = np.deg2rad(15)

        # Workspace bounds.
        self.pixel_size = 0.003125
        self.camera_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        # Only True for goal-based tasks, IF testing (affects ground-truth labels).
        self.goal_cond_testing = False

        # If we've taken identity action, exit gracefully (for bag tasks).
        self.exit_gracefully = False

    #-------------------------------------------------------------------------
    # Oracle Implementation
    #-------------------------------------------------------------------------

    def params_no_rots(self, vertex_pos, target_pos, overshoot):
        """Helper to handle common pick-place code for the oracle policy.

        We often have this patten: vertex positions and target positions in
        2D, and then potentially slightly overshoot the target. For example,
        with cloth it's helpful to do this since otherwise the physics will
        favor the cloth 'resetting' to its original state. Get the direction
        by creating vectors and then add to the new target. Then form tuples
        for the action params. Assumes no new rotations.

        Args:
            vertex_pos: 2D tuple or array representing picking position.
            target_pos: 2D tuple or array representing target position.
            overshoot: how much to go beyond the target position.

        Returns:
            Dict for the action with 'pose0' and 'pose1' keys.
        """
        p0 = (vertex_pos[0], vertex_pos[1], 0.001)
        p1 = (target_pos[0], target_pos[1], 0.001)
        direction = np.float32(p0) - np.float32(p1)
        length = np.linalg.norm(direction)
        direction = direction / length
        new_p0 = np.float32(p1) + direction * (length - 0.00)
        new_p1 = np.float32(p0) - direction * (length + overshoot)
        p0, p1 = tuple(new_p0), tuple(new_p1)
        params = {'pose0': (p0, (0,0,0,1)), 'pose1': (p1, (0,0,0,1))}
        return params

    def oracle(self, env):
        """Implement the oracle policy.

        Call task.oracle(env) to get a policy used to get demonstrations,
        defined in `act()`. Since this is not done within each task specific
        class, we assume a similar structure based on the type of action
        primitive used in the environment (one of three types).

        For some custom tasks, especially with bags, we may reach failures
        (e.g., if messing up the bag so badly). If our code doesn't work,
        return the identity action, which will usually not affect the task
        (as there's no translation) and with enough of these, we'll
        eventually hit the action limit.
        """
        OracleAgent = collections.namedtuple('OracleAgent', ['act'])

        def act(obs, info):
            act = {'camera_config': self.camera_config, 'primitive': None}
            if not obs or self.done():
                return act

            # Oracle uses ground truth object segmentation masks.
            colormap, heightmap, object_mask = self.get_object_masks(env)

            if (isinstance(self, tasks.names['cable-ring']) or
                    isinstance(self, tasks.names['cable-ring-notarget'])):
                # ------------------------------------------------------------ #
                # Instead of assigning fixed target zones, pick the closest
                # circle to target mapping, then correct the largest
                # discrepancy. Mostly the same as bag-alone-open; keeping these
                # separate for now just in case we change the envs. Only care
                # about positions here; orientation doesn't matter. Upon
                # inspection, I think we could fine-tune this a bit more by
                # avoiding any pull that makes the bead cross over itself. Not
                # sure how to precisely do that.
                # ------------------------------------------------------------ #
                vert_pos_l = []
                for bead_ID in self.cable_bead_IDs:
                    bead_position = p.getBasePositionAndOrientation(bead_ID)[0]
                    vert_pos_l.append(bead_position)
                vertpos_xyz_np = np.array(vert_pos_l)

                targets_l = []
                for bead_ID in self.goal['places']:
                    target_position, _ = self.goal['places'][bead_ID]
                    targets_l.append(target_position)
                targets_xyz_np = np.array(targets_l)

                assert vertpos_xyz_np.shape == targets_xyz_np.shape
                nb_maps = len(self.cable_bead_IDs)
                min_dist = np.float('inf')
                vertex_pos, target_pos = None, None

                # Different (but 'rotationally consistent') ordering of beads.
                for a in range(nb_maps * 2):
                    if a < nb_maps:
                        # mapping = [a, a+1, ..., nb_maps-1, 0, 1, ..., a-1]
                        mapping = [i for i in range(a, nb_maps)] + [i for i in range(0, a)]
                    else:
                        # Same as above but reverse it (to handle flipped ring).
                        a -= nb_maps
                        mapping = [i for i in range(a, nb_maps)] + [i for i in range(0, a)]
                        mapping = mapping[::-1]
                    differences = targets_xyz_np - vertpos_xyz_np[mapping]
                    distances = np.linalg.norm(differences, axis=1)
                    average_distance = np.mean(distances)

                    if average_distance < min_dist:
                        # Index of the largest distance among vertex + target.
                        max_idx = np.argmax(distances)
                        vertex_pos = vertpos_xyz_np[mapping][max_idx]
                        target_pos = targets_xyz_np[max_idx]
                        min_dist = average_distance

                overshoot = 0.0
                act['params'] = self.params_no_rots(vertex_pos, target_pos, overshoot)

            elif (isinstance(self, tasks.names['cloth-flat']) or
                    isinstance(self, tasks.names['cloth-flat-notarget'])):
                # ------------------------------------------------------------ #
                # Hopefully the easiest of all the cloth environments. Since
                # the cloth is always in the same side up, we can assume a
                # clockwise ordering of cloth corners to zone corners. The
                # first action is key and it should grip the cloth corner
                # closest to the zone and pull it to the furthest zone corner.
                # See: https://github.com/DanielTakeshi/pybullet-def-envs/pull/6
                # ------------------------------------------------------------ #
                mappings = [[0, 1, 2, 3],
                            [3, 0, 1, 2],
                            [2, 3, 0, 1],
                            [1, 2, 3, 0],]

                # Get cloth mesh data and info about cloth/zone corners.
                _, vert_pos_l = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
                corner_idx_np = np.array(self.corner_indices)
                targets_xy_np = self.corner_targets_xy
                vertpos_xy_np = np.array(vert_pos_l)[:, :2]
                min_dist = np.float('inf')
                min_std = np.float('inf')
                vertex_pos = None
                target_pos = None

                # Iterate through corner assignments.
                for mapping in mappings:
                    corners = corner_idx_np[mapping]
                    differences = targets_xy_np - vertpos_xy_np[corners]
                    distances = np.linalg.norm(differences, axis=1)
                    avg_dist = np.min(distances)
                    avg_std = np.std(distances)

                    if (self.t == 0) and (avg_std <= min_std):
                        # Pick cloth corner closest to _zone_ center.
                        min_std = avg_std
                        zone_xy = np.array(self.zone_pose[0][:2]).reshape(1,2)
                        differences = zone_xy - vertpos_xy_np[corners]
                        distances = np.linalg.norm(differences, axis=1)
                        idx = np.argmin(distances)
                        vertex_pos = vertpos_xy_np[corners][idx]
                        target_pos = targets_xy_np[idx]
                    elif (self.t != 0) and (avg_dist <= min_dist):
                        # Otherwise, consider largest discrepancy in match.
                        min_dist = avg_dist
                        idx = np.argmax(distances)
                        vertex_pos = vertpos_xy_np[corners][idx]
                        target_pos = targets_xy_np[idx]
                    else:
                        # If the above don't apply, DON'T update positions.
                        pass
                self.t += 1

                # Currently overshooting slightly due to cloth physics.
                overshoot = 0.03
                act['params'] = self.params_no_rots(vertex_pos, target_pos, overshoot)

            elif isinstance(self, tasks.names['cloth-cover']):
                # ------------------------------------------------------------ #
                # Hopefully a straightforward one: put item on the center of
                # the cloth, then adjust cloth so it covers it using a
                # triangular fold. TODO: should we have a guard against failure
                # cases? I'm using task stages here instead of t=0 vs t=1 just
                # in case the first pick and place missed the cube, or in case
                # we have multiple cubes.
                # ------------------------------------------------------------ #
                assert len(self.block_IDs) == 1 # For now

                # Get cloth mesh data and info about cloth/zone corners.
                _, vert_pos_l = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
                corner_idx_np = np.array(self.corner_indices)
                vertpos_xy_np = np.array(vert_pos_l)[:, :2]
                cloth_center_xy = np.mean(vertpos_xy_np[corner_idx_np], axis=0)

                # Compute task stage. TODO this is a really bad hack.
                if self.t > 0:
                    self.task_stage = 2
                self.t += 1

                if self.task_stage == 1:
                    # Put a cube on the center of the cloth.
                    block_id = self.block_IDs[0]
                    vertex_pos = p.getBasePositionAndOrientation(block_id)[0]
                    target_pos = cloth_center_xy
                    overshoot = 0.0

                elif self.task_stage == 2:
                    # Fold the cloth. Must pick one of the four directions.
                    direction = np.random.randint(4)
                    if direction == 0:
                        source = corner_idx_np[0]
                        target = corner_idx_np[2]
                    elif direction == 1:
                        source = corner_idx_np[1]
                        target = corner_idx_np[3]
                    elif direction == 2:
                        source = corner_idx_np[2]
                        target = corner_idx_np[0]
                    elif direction == 3:
                        source = corner_idx_np[3]
                        target = corner_idx_np[1]
                    vertex_pos = vertpos_xy_np[source]
                    target_pos = vertpos_xy_np[target]
                    overshoot = 0.03

                # We adjusted overshooting earlier based on the task stage.
                act['params'] = self.params_no_rots(vertex_pos, target_pos, overshoot)

            elif isinstance(self, tasks.names['bag-alone-open']):
                # ------------------------------------------------------------ #
                # We have a circular ring of targets on the 2D plane, and want
                # to maximize bag opening area. If we have 32 'top ring
                # indices' then there are 32 targets. But due to rotations,
                # they can rotate, so there's 32 different 'assignments'. We
                # want to pick the one that is closest w.r.t. the xy plane.
                # THEN find the single bag top ring vertex and corner target
                # that's furthest. Somewhat naive but it's a start.
                # ------------------------------------------------------------ #
                # The main failure case is likely with the bag hiding its own
                # top ring, so I'm going to carefully set the starting state.
                # ------------------------------------------------------------ #
                # Unlike cable-ring envs, we do NOT have to consider "flipped
                # beads" in the rotation, due to the way we sample the bag.
                # ------------------------------------------------------------ #

                # Check object_mask for all IDs that correspond to the cable ring.
                cable_IDs = np.array(self.cable_bead_IDs)
                bead_mask = np.isin(object_mask, test_elements=cable_IDs) # :D

                # Detect visible beads. Is there a faster way to do this?
                visible_beads = []
                for bead in self.cable_bead_IDs:
                    if bead in object_mask:
                        visible_beads.append(bead)
                frac_visible = len(visible_beads) / len(self.cable_bead_IDs)

                # If only a few beads are visible, exit early.
                if frac_visible <= BEAD_THRESH:
                    self.exit_gracefully = True
                    print(f'WARNING: fraction of visible beads: {frac_visible}')
                    act['params'] = IDENTITY
                    act['primitive'] = self.primitive
                    return act

                # Now do bag-opening. TODO: only pick at VISIBLE beads?
                vert_pos_l = []
                for bead_ID in self.cable_bead_IDs:
                    bead_position = p.getBasePositionAndOrientation(bead_ID)[0]
                    vert_pos_l.append(bead_position)
                vertpos_xyz_np = np.array(vert_pos_l)

                targets_l = self.circle_target_positions
                targets_xyz_np = np.array( [[p[0],p[1],p[2]] for p in targets_l] )

                assert vertpos_xyz_np.shape == targets_xyz_np.shape
                assert len(self.top_ring_idxs) == len(self.cable_bead_IDs)
                nb_maps = len(self.top_ring_idxs)
                min_dist = np.float('inf')
                vertex_pos, target_pos = None, None

                for a in range(nb_maps):
                    # mapping = [a, a+1, ..., nb_maps-1, 0, 1, ..., a-1]
                    mapping = [i for i in range(a, nb_maps)] + [i for i in range(0, a)]
                    differences = targets_xyz_np - vertpos_xyz_np[mapping]
                    distances = np.linalg.norm(differences, axis=1)
                    average_distance = np.mean(distances)

                    if average_distance < min_dist:
                        # Index of the largest distance among vertex + target.
                        max_idx = np.argmax(distances)
                        vertex_pos = vertpos_xyz_np[mapping][max_idx]
                        target_pos = targets_xyz_np[max_idx]
                        min_dist = average_distance

                # Make the robot 'overshoot' slightly towards the target position.
                overshoot = 0.02
                act['params'] = self.params_no_rots(vertex_pos, target_pos, overshoot)

            elif (isinstance(self, tasks.names['bag-items-easy']) or
                    isinstance(self, tasks.names['bag-items-hard'])):
                # ------------------------------------------------------------ #
                # Hard-coding several stages for the task. We allocate task
                # selection to the task to enable both oracle and learned
                # policy adjust action parameters based on the task stage.
                # ------------------------------------------------------------ #
                # (1) Do what we did for bag-alone-open and naively ignore z
                # coordinate, to get the area of the convex hull.
                # ------------------------------------------------------------ #
                # (2) For this stage, assume we were successful, because it
                # almost always works, and we have code that supports dropping
                # a rigid item onto a deformable. Refer to `self.item_IDs` for
                # items we have to insert. Use the segmentation mask to figure
                # out a good placing location.
                # ------------------------------------------------------------ #
                # (3) Move the bag with the item(s) by picking at a visible bead.
                # ------------------------------------------------------------ #

                # Detect visible beads. Is there a faster way to do this?
                visible_beads = []
                for bead in self.cable_bead_IDs:
                    if bead in object_mask:
                        visible_beads.append(bead)

                # Allocate the stage selection to the task-specific method.
                success, place_pixels_eroded = self.determine_task_stage(
                    colormap=colormap,
                    heightmap=heightmap,
                    object_mask=object_mask,
                    visible_beads=visible_beads,
                )

                # Exit gracefully if no success OR if we determined we should in reset().
                if (not success) or (self.exit_gracefully):
                    self.exit_gracefully = True  # adding here to handle `not success`
                    act['params'] = IDENTITY
                    act['primitive'] = self.primitive
                    return act

                # ------------------------------ #
                # Perform task-dependent actions #
                # ------------------------------ #

                if self.task_stage == 1:
                    # Copy bag-alone-open. TODO: only pick at VISIBLE beads?
                    vert_pos_l = []
                    for bead_ID in self.cable_bead_IDs:
                        bead_position = p.getBasePositionAndOrientation(bead_ID)[0]
                        vert_pos_l.append(bead_position)
                    vertpos_xyz_np = np.array(vert_pos_l)

                    targets_l = self.circle_target_positions
                    targets_xyz_np = np.array( [[p[0],p[1],p[2]] for p in targets_l] )

                    assert vertpos_xyz_np.shape == targets_xyz_np.shape
                    assert len(self.top_ring_idxs) == len(self.cable_bead_IDs)
                    nb_maps = len(self.top_ring_idxs)
                    min_dist = np.float('inf')
                    vertex_pos, target_pos = None, None

                    for a in range(nb_maps):
                        # mapping = [a, a+1, ..., nb_maps-1, 0, 1, ..., a-1]
                        mapping = [i for i in range(a, nb_maps)] + [i for i in range(0, a)]
                        differences = targets_xyz_np - vertpos_xyz_np[mapping]
                        distances = np.linalg.norm(differences, axis=1)
                        average_distance = np.mean(distances)

                        if average_distance < min_dist:
                            # Index of the largest distance among vertex + target.
                            max_idx = np.argmax(distances)
                            vertex_pos = vertpos_xyz_np[mapping][max_idx]
                            target_pos = targets_xyz_np[max_idx]
                            min_dist = average_distance

                    # Make the robot 'overshoot' slightly towards the target position.
                    overshoot = 0.02
                    act['params'] = self.params_no_rots(vertex_pos, target_pos, overshoot)

                elif self.task_stage == 2:
                    # I ran into this failure. Does it only happen with 3.0.4 w/out disp?
                    if np.sum(np.float32(place_pixels_eroded)) == 0:
                        print('Warning, place_pixels_eroded has no pixels left.')
                        self.save_images(colormap, heightmap, object_mask)
                        self.exit_gracefully = True
                        act['params'] = IDENTITY
                        act['primitive'] = self.primitive
                        return act

                    # Identify an item that is not within the bag.
                    item = None
                    for ID in self.item_IDs:
                        if ID not in self.items_in_bag_IDs:
                            item = ID
                            pick_mask = np.uint8(object_mask == item)
                            pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

                            # This should not happen, so let's exit gracefully if it does.
                            # Or does it only apply if using PyBullet 3.0.4?
                            if np.sum(pick_mask) == 0:
                                print('Something bad happened, pick mask sum is 0?')
                                self.save_images(colormap, heightmap, object_mask, pick_mask)
                                self.exit_gracefully = True
                                act['params'] = IDENTITY
                                act['primitive'] = self.primitive
                                return act
                            break
                    assert item is not None

                    # Key assumption: we assume the placing will be successful.
                    self.items_in_bag_IDs.append(item)

                    # Compute picking point. Sample anywhere on item's eroded area.
                    pick_prob = np.float32(pick_mask)
                    pick_pixel = utils.sample_distribution(pick_prob)
                    pick_position = utils.pixel_to_position(
                            pick_pixel, heightmap, self.bounds, self.pixel_size)
                    p0 = pick_position

                    # Placing position. Sample anywhere on the open eroded area.
                    place_prob = np.float32(place_pixels_eroded)
                    place_pixel = utils.sample_distribution(place_prob)
                    place_position = utils.pixel_to_position(
                            place_pixel, heightmap, self.bounds, self.pixel_size)
                    p1 = place_position

                    # Get the usual action parameter, without overshooting.
                    act['params'] = self.params_no_rots(p0, p1, overshoot=0.0)

                    if isinstance(self, tasks.names['bag-items-hard']):
                        new_pose1 = act['params']['pose1']

                        # But now sample the rotation, assuming we use 24 (TODO: make more robust?).
                        num_rots = 24
                        _rots = [i * 2 * np.pi / num_rots for i in range(num_rots)]
                        _rot = np.random.choice(_rots)
                        _rot_deg = _rot*180/np.pi
                        print(f'Sampled rotation: {_rot_deg:0.4f}')

                        # Assign placing pose. Picking still uses identity rotation.
                        new_rot1 = p.getQuaternionFromEuler((0, 0, _rot))
                        new_pose1 = (new_pose1[0], new_rot1)
                        act['params']['pose1'] = new_pose1

                elif self.task_stage == 3:
                    # Bag gripping + depositing. Currently considering any VISIBLE bead as
                    # pick points at random. If we filter data, hopefully a pattern appears.
                    p0 = None
                    p1 = self.zone_pose[0]

                    # This should never happen but just in case ...
                    if len(visible_beads) == 0:
                        print(f'WARNING: no visible beads in task stage 3??')
                        visible_beads.append(self.cable_bead_IDs[0])

                    bead_ID = np.random.choice(visible_beads)
                    p0 = p.getBasePositionAndOrientation(bead_ID)[0]
                    act['params'] = self.params_no_rots(p0, p1, overshoot=0.0)

            elif isinstance(self, tasks.names['bag-color-goal']):
                # ------------------------------------------------------------ #
                # Open a target color bag and move the item to the target.
                # NOTE: use cable_bead_target_bag_IDs, not cable_bead_IDs
                # ------------------------------------------------------------ #

                # Detect visible beads. Is there a faster way to do this?
                visible_beads = []
                for bead in self.cable_bead_target_bag_IDs:
                    if bead in object_mask:
                        visible_beads.append(bead)

                # Allocate the stage selection to the task-specific method.
                success, place_pixels_eroded = self.determine_task_stage(
                    colormap=colormap,
                    heightmap=heightmap,
                    object_mask=object_mask,
                    visible_beads=visible_beads,
                )

                # Exit gracefully if no success OR if we determined we should in reset().
                if (not success) or (self.exit_gracefully):
                    self.exit_gracefully = True  # adding here to handle `not success`
                    act['params'] = IDENTITY
                    act['primitive'] = self.primitive
                    return act

                # ------------------------------ #
                # Perform task-dependent actions #
                # ------------------------------ #

                if self.task_stage == 1:
                    # Copy bag-alone-open. TODO: only pick at VISIBLE beads?
                    vert_pos_l = []
                    for bead_ID in self.cable_bead_target_bag_IDs:
                        bead_position = p.getBasePositionAndOrientation(bead_ID)[0]
                        vert_pos_l.append(bead_position)
                    vertpos_xyz_np = np.array(vert_pos_l)

                    targets_l = self.circle_target_positions
                    targets_xyz_np = np.array( [[p[0],p[1],p[2]] for p in targets_l] )

                    assert vertpos_xyz_np.shape == targets_xyz_np.shape
                    assert len(self.top_ring_idxs) == len(self.cable_bead_target_bag_IDs)
                    nb_maps = len(self.top_ring_idxs)
                    min_dist = np.float('inf')
                    vertex_pos, target_pos = None, None

                    for a in range(nb_maps):
                        # mapping = [a, a+1, ..., nb_maps-1, 0, 1, ..., a-1]
                        mapping = [i for i in range(a, nb_maps)] + [i for i in range(0, a)]
                        differences = targets_xyz_np - vertpos_xyz_np[mapping]
                        distances = np.linalg.norm(differences, axis=1)
                        average_distance = np.mean(distances)

                        if average_distance < min_dist:
                            # Index of the largest distance among vertex + target.
                            max_idx = np.argmax(distances)
                            vertex_pos = vertpos_xyz_np[mapping][max_idx]
                            target_pos = targets_xyz_np[max_idx]
                            min_dist = average_distance

                    # Make the robot 'overshoot' slightly towards the target position.
                    overshoot = 0.02
                    act['params'] = self.params_no_rots(vertex_pos, target_pos, overshoot)

                elif self.task_stage == 2:
                    # I ran into this failure. Does it only happen with 3.0.4 w/out disp?
                    if (place_pixels_eroded is None) or (np.sum(np.float32(place_pixels_eroded)) == 0):
                        print('Warning, place_pixels_eroded has no pixels left or is None.')
                        self.save_images(colormap, heightmap, object_mask)
                        self.exit_gracefully = True
                        act['params'] = IDENTITY
                        act['primitive'] = self.primitive
                        return act

                    # ONLY do the target item.
                    item = self.single_block_ID
                    pick_mask = np.uint8(object_mask == item)
                    pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

                    # This should not happen, so let's exit gracefully if it does.
                    # Or does it only apply if using PyBullet 3.0.4?
                    if np.sum(pick_mask) == 0:
                        print('Something bad happened, pick mask sum is 0?')
                        self.save_images(colormap, heightmap, object_mask, pick_mask)
                        self.exit_gracefully = True
                        act['params'] = IDENTITY
                        act['primitive'] = self.primitive
                        return act

                    # Compute picking point. Sample anywhere on item's eroded area.
                    pick_prob = np.float32(pick_mask)
                    pick_pixel = utils.sample_distribution(pick_prob)
                    pick_position = utils.pixel_to_position(
                            pick_pixel, heightmap, self.bounds, self.pixel_size)
                    p0 = pick_position

                    # Placing position. Sample anywhere on the open eroded area.
                    place_prob = np.float32(place_pixels_eroded)
                    place_pixel = utils.sample_distribution(place_prob)
                    place_position = utils.pixel_to_position(
                            place_pixel, heightmap, self.bounds, self.pixel_size)
                    p1 = place_position

                    # Get the usual action parameter, without overshooting.
                    act['params'] = self.params_no_rots(p0, p1, overshoot=0.0)
                    new_pose1 = act['params']['pose1']

                    # Assign placing pose. Picking still uses identity rotation.
                    new_rot1 = p.getQuaternionFromEuler((0, 0, 0))
                    new_pose1 = (new_pose1[0], new_rot1)
                    act['params']['pose1'] = new_pose1

            elif self.primitive == 'pick_place':
                # Trigger reset if no ground truth steps are available.
                if len(self.goal['steps']) == 0:
                    self.goal['steps'] = []  # trigger done then reset
                    return act

                # Get possible picking locations (prioritize furthest).
                next_step = self.goal['steps'][0]
                possible_objects = np.int32(list(next_step.keys())).copy()
                distances = []
                for object_id in possible_objects:
                    position = p.getBasePositionAndOrientation(object_id)[0]
                    targets = next_step[object_id][1]
                    targets = [t for t in targets if t in self.goal['places']]
                    places = [self.goal['places'][t][0] for t in targets]
                    d = np.float32(places) - np.float32(position).reshape(1, 3)
                    distances.append(np.min(np.linalg.norm(d, axis=1)))

                distances_sort = np.argsort(distances)[::-1]
                possible_objects = possible_objects[distances_sort]
                for object_id in possible_objects:
                    pick_mask = np.uint8(object_mask == object_id)
                    pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))
                    if np.sum(pick_mask) > 0:
                        break

                # Trigger task reset if no object is visible.
                if np.sum(pick_mask) == 0:
                    self.goal['steps'] = []  # trigger done then reset
                    return act

                # Compute picking pose.
                pick_prob = np.float32(pick_mask)
                pick_pixel = utils.sample_distribution(pick_prob)
                pick_position = utils.pixel_to_position(
                    pick_pixel, heightmap, self.bounds, self.pixel_size)
                pick_rotation = p.getQuaternionFromEuler((0, 0, 0))
                pick_pose = (pick_position, pick_rotation)

                # Get candidate target placing poses.
                targets = next_step[object_id][1]
                targets = [pi for pi in targets if pi in self.goal['places']]
                i = np.random.randint(0, len(targets))
                true_pose = self.goal['places'][targets[i]]

                # Compute placing pose.
                object_pose = p.getBasePositionAndOrientation(object_id)
                world_to_pick = self.invert(pick_pose)
                object_to_pick = self.multiply(world_to_pick, object_pose)
                pick_to_object = self.invert(object_to_pick)
                place_pose = self.multiply(true_pose, pick_to_object)

                # For various cable tasks, we don't want to apply rotations.
                if (isinstance(self, tasks.names['cable']) or
                    isinstance(self, tasks.names['cable-shape']) or
                    isinstance(self, tasks.names['cable-shape-notarget']) or
                    isinstance(self, tasks.names['cable-line-notarget'])):
                    place_pose = (place_pose[0], (0, 0, 0, 1))

                params = {'pose0': pick_pose, 'pose1': place_pose}
                act['params'] = params

            elif isinstance(self, tasks.names['sweeping']):
                p0 = None
                p1 = self.zone_pose[0]

                # Set farthest object position as start position.
                for object_id in self.object_points:
                    object_pose = p.getBasePositionAndOrientation(object_id)
                    position = self.object_points[object_id].squeeze()
                    position = self.apply(object_pose, position)
                    d = np.linalg.norm(np.float32(position) - np.float32(p1))
                    if (p0 is None) or (d > threshold):
                        p0 = position
                        threshold = d

                # Adjust start and end positions.
                p0 = (p0[0], p0[1], 0.001)
                p1 = (p1[0], p1[1], 0.001)
                rotation = p.getQuaternionFromEuler((0, 0, 0))
                direction = np.float32(p0) - np.float32(p1)
                length = np.linalg.norm(direction)
                direction = direction / length
                new_p0 = np.float32(p1) + direction * (length + 0.02)
                new_p1 = np.float32(p0) - direction * (length - 0.05)
                p0, p1 = tuple(new_p0), tuple(new_p1)
                params = {'pose0': (p0, rotation), 'pose1': (p1, rotation)}
                act['params'] = params

            elif isinstance(self, tasks.names['pushing']):
                # Get start position.
                p0 = np.float32(p.getLinkState(env.ur5, env.ee_tip_link)[0])
                rotation = p.getQuaternionFromEuler((0, 0, 0))

                # Compute end position.
                goal_position = np.array([0.5, -0.5, 0])
                object_id = env.objects[0]
                object_pose = p.getBasePositionAndOrientation(object_id)
                world_to_object = self.invert(object_pose)
                goal_position = self.apply(world_to_object, goal_position)
                p1_object = np.float32(goal_position)
                p1_object[0] = -p1_object[0] * 2
                p1 = self.apply(object_pose, p1_object)
                push_direction = (p1 - p0) / np.linalg.norm((p1 - p0))
                p1 = p0 + push_direction * 0.01
                params = {'pose0': (p0, rotation), 'pose1': (p1, rotation)}
                act['params'] = params
            else:
                raise ValueError('Tried to define action, but task {} is not '
                    'supported:\n{}'.format(self, tasks.names))

            act['primitive'] = self.primitive
            return act

        return OracleAgent(act)

    #-------------------------------------------------------------------------
    # Reward Function and Task Completion Metrics
    #-------------------------------------------------------------------------

    def reward(self):
        """Compute reward for current timestep.

        Default Ravens assumes "delta rewards" that start at 0, and only
        reach 1 if all steps worked. For our environments, the most natural
        reward might (a) not start at 0, and/or (b) might be OK with
        something below 1. However, to simplify things, keep
        `self.total_rewards` as whatever I want to measure (e.g., coverage,
        ring area, etc.) and `reward` as the delta, BUT when EVALUATING, make
        sure I use stuff from `info[extras]` for more refined analysis.

        Returns:
            (reward, extras): tuple of the DELTA reward, plus extra
                information. I return an extra dictionary to provide
                additional information about the reward, e.g., the raw
                coverage values (before the 'deltas' are computed for the
                actual reward). Again, `reward` is the DELTA reward.
        """
        reward = 0
        extras = {}

        # Careful! This condition confused me about cloth-cover XD.
        if self.done():
            return reward, extras

        # Pose-based evaluation metric.
        if self.metric == 'pose':
            curr_step = self.goal['steps'][0]  # pass-by-reference

            for object_id in list(curr_step.keys()):
                curr_pose = p.getBasePositionAndOrientation(object_id)

                # Get all possible placement poses.
                places_positions = np.zeros((0, 3))
                places_rotations = np.zeros((0, 3))
                symmetry, places = curr_step[object_id]
                places = [t for t in places if t in self.goal['places']]
                for place in places:
                    pose = self.goal['places'][place]
                    places_positions = np.vstack((places_positions, pose[0]))
                    rotation = p.getEulerFromQuaternion(pose[1])
                    places_rotations = np.vstack((places_rotations, rotation))

                # Compute translational error.
                curr_position = np.array(curr_pose[0])[:2].reshape(1, 2)
                error_t = places_positions[:, :2] - curr_position
                error_t = np.linalg.norm(error_t, axis=1)

                # Compute rotational error.
                error_r = 0
                if symmetry > 0:
                    curr_rotation = p.getEulerFromQuaternion(curr_pose[1])[2]
                    error_r = places_rotations[:, 2] - curr_rotation
                    error_r = abs(error_r) % symmetry
                    neg_ind = error_r > (symmetry / 2)
                    error_r[neg_ind] = symmetry - error_r[neg_ind]

                # Compute reward from error.
                success_t = error_t < self.position_eps
                success_r = error_r < self.rotation_eps
                success = success_t & success_r
                if any(success):
                    reward += 1. / self.num_steps

                    # Remove from possible placement poses.
                    place = places[np.argwhere(success).squeeze()]
                    self.goal['places'].pop(place)
                    curr_step.pop(object_id)

                    # Next step?
                    if len(curr_step) == 0:
                        self.goal['steps'].pop(0)

        # Zone-based evaluation metric.
        elif self.metric == 'zone':
            # Daniel: get the inverse mapping so we go from object to zone,
            # that way we can directly use self.zone_size for testing bounds.
            total_rewards = 0
            zone_points = []
            for object_id in self.object_points:
                points = self.object_points[object_id]
                object_pose = p.getBasePositionAndOrientation(object_id)
                world_to_zone = self.invert(self.zone_pose)
                object_to_zone = self.multiply(world_to_zone, object_pose)
                points = np.float32(self.apply(object_to_zone, points))
                # For cable env, valid_points has info about one bead.
                valid_points = np.logical_and.reduce([
                    points[0, :] > -self.zone_size[0] / 2,
                    points[0, :] < self.zone_size[0] / 2,
                    points[1, :] > -self.zone_size[1] / 2,
                    points[1, :] < self.zone_size[1] / 2,
                    points[2, :] > -0.01,
                    points[2, :] < self.bounds[2, 1]]).tolist()
                if hasattr(self, 'goal'):
                    if not isinstance(self, tasks.names['cable']):
                        if len(self.goal['steps']) and any(valid_points):
                            if object_id == list(self.goal['steps'][0].keys())[0]:
                                self.goal['steps'].pop(0)
                zone_points += valid_points
            total_rewards = np.sum(np.array(zone_points)) / len(zone_points)
            reward = total_rewards - self.total_rewards
            self.total_rewards = total_rewards

            # Palletizing: spawn another box in the workspace if it is empty.
            if isinstance(self, tasks.names['palletizing']):
                if len(self.goal['steps']) > 0:
                    workspace_empty = True
                    for object_id in self.object_points:
                        object_pose = p.getBasePositionAndOrientation(
                            object_id)
                        workspace_empty = workspace_empty and (
                            (object_pose[0][1] < -0.5) or (object_pose[0][1] > 0))
                    if workspace_empty:
                        object_id = list(self.goal['steps'][0].keys())[0]
                        theta = np.random.random() * 2 * np.pi
                        rotation = p.getQuaternionFromEuler((0, 0, theta))
                        p.resetBasePositionAndOrientation(
                            object_id, [0.5, -0.25, 0.1], rotation)

        elif self.metric == 'cable-target':
            # ---------------------------------------------------------------- #
            # When a cable has to match a target, but where the target is not
            # necessarily a straight line; for a straight line, we could use
            # `zone_size` but here it's easier to iterate through all possible
            # targets. ONLY consider those IDs in `cable_bead_IDs`. Actually we
            # don't need the machinery with poses because `object_points`
            # should be zero for the beads. Note that we only check if the bead
            # is close to ANY target. This resolves ambiguity if the cable is
            # reversed, or with a ring of cables. Note: 2*radius is too strict
            # of a threshold. While I briefly tried the 3D distance between
            # beads and targets 2D empirically works well.
            # ---------------------------------------------------------------- #
            total_rewards = 0
            dist_threshold = self.radius * 3.5 # Try this? 2*radius is too strict.
            zone_points = 0
            for bead_id in self.cable_bead_IDs:
                bead_position = p.getBasePositionAndOrientation(bead_id)[0]
                min_d = np.float('inf')
                for key in self.goal['places']:
                    target_position = self.goal['places'][key][0]
                    bead_xy = np.array([bead_position[0], bead_position[1]])
                    target_xy = np.array([target_position[0], target_position[1]])
                    dist = np.linalg.norm(bead_xy - target_xy)
                    if dist < min_d:
                        min_d = dist
                if min_d < dist_threshold:
                    zone_points += 1
            total_rewards = zone_points / len(self.cable_bead_IDs)
            reward = total_rewards - self.total_rewards
            self.total_rewards = total_rewards

            # Helps us see if performance varies based on the target property.
            extras['nb_sides'] = self.nb_sides
            extras['nb_beads'] = len(self.cable_bead_IDs)
            extras['nb_zone'] = zone_points
            extras['delta_reward'] = reward
            extras['total_rewards'] = total_rewards

        elif self.metric == 'cable-ring':
            # ---------------------------------------------------------------- #
            # We should measure the convex hull of the area enclosed by the
            # beads. Naively ignore z coordinates so we just project onto
            # (x,y). Assumes just one ring specified by `self.cable_bead_IDs`.
            # Ignore the delta rewards because those are deltas of convex hulls
            # and not easily interpretable. Just look at `self.total_rewards`
            # at any time and see if it clears a fraction.
            # ---------------------------------------------------------------- #
            bead_IDs = self.cable_bead_IDs
            points = []
            for bead_ID in self.cable_bead_IDs:
                bead_position = p.getBasePositionAndOrientation(bead_ID)[0]
                points.append( [bead_position[0], bead_position[1]] )
            points = np.array([ [p[0],p[1]] for p in points ])

            # In 2D, this returns AREA (hull.area returns perimeter).
            try:
                hull = ConvexHull(points)
                convex_hull_area = hull.volume
            except scipy.spatial.qhull.QhullError as e:
                print(e)
                convex_hull_area = 0
            total_rewards = convex_hull_area
            reward = total_rewards - self.total_rewards
            self.total_rewards = total_rewards

            # `total_rewards` is redundant here but keep for consistency
            extras['convex_hull_area'] = convex_hull_area
            extras['best_possible_area'] = self.circle_area
            extras['fraction'] = convex_hull_area / self.circle_area
            extras['delta_reward'] = reward
            extras['total_rewards'] = total_rewards

        elif self.metric == 'cloth-coverage':
            # ---------------------------------------------------------------- #
            # Since we have an arbitrary target zone, using the convex hull (as
            # in our IROS 2020 paper) is insufficient because we might have
            # flattened the cloth in the wrong area. Use pixel-based coverage.
            # ---------------------------------------------------------------- #
            IoU, coverage = self.compute_pixel_IoU_coverage()

            # The usual delta-based metrics. Use coverage but it could be IoU...
            total_rewards = coverage
            reward = total_rewards - self.total_rewards
            self.total_rewards = total_rewards

            extras['cloth_IoU'] = IoU
            extras['cloth_coverage'] = coverage
            extras['reward'] = reward
            extras['total_rewards'] = total_rewards

        elif self.metric == 'cloth-cover-item':
            # ---------------------------------------------------------------- #
            # Cover an item with cloth. Current solution: get object mask and
            # see if the item is there. Additional sanity check: that the item
            # is actually within some distance from the cloth, otherwise it
            # could have fallen outside the workspace.
            # ---------------------------------------------------------------- #
            _, vert_pos_l = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
            corner_idx_np = np.array(self.corner_indices)
            vertpos_xy_np = np.array(vert_pos_l)[:, :2]
            cloth_center_xy = np.mean(vertpos_xy_np[corner_idx_np], axis=0)

            # Is the cube close to the center of the cloth by a threshold?
            assert len(self.block_IDs) == 1, self.block_IDs
            block_id = self.block_IDs[0]
            block = p.getBasePositionAndOrientation(block_id)[0]
            dist_block2cent = np.linalg.norm(block[:2] - cloth_center_xy)

            # Get segmentation condtion with the distance condition.
            is_item_covered = self.is_item_covered()
            if dist_block2cent > 0.25:
                total_rewards = 0
            else:
                total_rewards = is_item_covered
            reward = total_rewards - self.total_rewards
            self.total_rewards = total_rewards

            extras['dist_block2cent'] = dist_block2cent
            extras['is_item_covered'] = is_item_covered
            extras['total_rewards'] = self.total_rewards

        elif self.metric == 'bag-alone-open':
            # ---------------------------------------------------------------- #
            # Measure area of the 'bag opening' visible. Given that we're doing
            # top-down pick and place (instead of 6 DoF) then it probably makes
            # sense (for now) just to measure the 2D projection of the top ring
            # vertices to the plane, and get the area from that? TODO: we need
            # to include a test for VERTEX VISIBILITY.
            # ---------------------------------------------------------------- #
            bead_IDs = self.cable_bead_IDs
            points = []
            for bead_ID in self.cable_bead_IDs:
                bead_position = p.getBasePositionAndOrientation(bead_ID)[0]
                points.append( [bead_position[0], bead_position[1]] )
            points = np.array([ [p[0],p[1]] for p in points ])

            # In 2D, this returns AREA (hull.area returns perimeter).
            try:
                hull = ConvexHull(points)
                convex_hull_area = hull.volume
            except scipy.spatial.qhull.QhullError as e:
                print(e)
                convex_hull_area = 0
            total_rewards = convex_hull_area
            reward = total_rewards - self.total_rewards
            self.total_rewards = total_rewards

            # `total_rewards` is redundant here but keep for consistency
            extras['convex_hull_area'] = convex_hull_area
            extras['best_possible_area'] = self.circle_area
            extras['fraction'] = convex_hull_area / self.circle_area
            extras['delta_reward'] = reward
            extras['total_rewards'] = self.total_rewards

        elif self.metric == 'bag-items':
            # ---------------------------------------------------------------- #
            # First, we give a small positive reward conditioned upon reaching
            # task stage 3. We ignore rewards in the earlier tasks because (a)
            # items might start in the zone already, and (b) the bag might
            # already be open. It's imperfect but reasonable. Then for task 3,
            # it consists of the percentage of beads (of bag) in the zone and
            # if the cube is in the zone, with more weight to the cube.
            # Currently we're not checking if the cube has been raised in
            # height during the pull. Track stuff in `extras`.
            # ---------------------------------------------------------------- #
            total_rewards = 0

            def points_in_zone(object_id):
                # For beads / small cubes, this is a binary test: 'is it in zone?'
                # Bigger blocks should have multiple points per object.
                points = self.object_points[object_id]
                object_pose = p.getBasePositionAndOrientation(object_id)
                world_to_zone = self.invert(self.zone_pose)
                object_to_zone = self.multiply(world_to_zone, object_pose)
                points = np.float32(self.apply(object_to_zone, points))
                valid_points = np.logical_and.reduce([
                    points[0, :] > -self.zone_size[0] / 2,
                    points[0, :] < self.zone_size[0] / 2,
                    points[1, :] > -self.zone_size[1] / 2,
                    points[1, :] < self.zone_size[1] / 2,
                    points[2, :] > -0.01,
                    points[2, :] < self.bounds[2, 1]]).tolist()
                return valid_points

            zone_items_rew = 0
            zone_beads_rew = 0

            if self.task_stage == 3:
                # 50% weight: items we actually want to be in the zone.
                zone_items = []
                for object_id in self.item_IDs:
                    valid_points = points_in_zone(object_id)
                    zone_items += valid_points
                zone_items_rew = np.sum(np.array(zone_items)) / len(zone_items)
                zone_items_rew *= 0.5

                # 50% weight: the cable beads.
                zone_beads = []
                for bead_id in self.cable_bead_IDs:
                    valid_points = points_in_zone(bead_id)
                    zone_beads += valid_points
                zone_beads_rew = np.sum(np.array(zone_beads)) / len(zone_beads)
                zone_beads_rew *= 0.5

            # Get total_rewards, then the usual delta and self.total_rewards.
            total_rewards = zone_items_rew + zone_beads_rew
            reward = total_rewards - self.total_rewards
            self.total_rewards = total_rewards

            # Track the convex hull by copying bag-open-alone.
            bead_IDs = self.cable_bead_IDs
            points = []
            for bead_ID in self.cable_bead_IDs:
                bead_position = p.getBasePositionAndOrientation(bead_ID)[0]
                points.append( [bead_position[0], bead_position[1]] )
            points = np.array([ [p[0],p[1]] for p in points ])
            try:
                hull = ConvexHull(points)
                convex_hull_area = hull.volume
            except scipy.spatial.qhull.QhullError as e:
                print(e)
                convex_hull_area = 0

            # Track the `self.task_stage`, particularly important for these envs.
            extras['convex_hull_area'] = convex_hull_area
            extras['best_possible_area'] = self.circle_area
            extras['reward'] = reward
            extras['total_rewards'] = self.total_rewards
            extras['task_stage'] = self.task_stage
            extras['zone_items_rew'] = zone_items_rew
            extras['zone_beads_rew'] = zone_beads_rew

        elif self.metric == 'bag-color-goal':
            total_rewards = 0

            # Get segmentation conditions. See the method for further details.
            result_info = self.is_item_in_bag()

            # (Oct 27) fraction is 0.25, sometimes I observe part of distract inside the
            # hull but if we really get the correct item inside, then maybe it's OK?
            if result_info['exit_early']:
                total_rewards = 0
            else:
                total_rewards = result_info['frac_in_target_bag']

            # The usual delta and self.total_rewards.
            reward = total_rewards - self.total_rewards
            self.total_rewards = total_rewards

            extras['frac_in_target_bag']   = result_info['frac_in_target_bag']
            extras['frac_in_distract_bag'] = result_info['frac_in_distract_bag']
            extras['bag_color_target']     = self.bag_colors[0]  # remember, 0 index is target
            extras['bag_color_distract']   = self.bag_colors[1]  # for now just one other bag
            extras['reward']               = reward
            extras['total_rewards']        = self.total_rewards
            extras['task_stage']           = self.task_stage

        else:
            raise NotImplementedError(self.metric)

        return reward, extras

    def done(self):
        """Check if the task is done AND has not failed.

        To be clear: for normal Ravens envs, `self.total_rewards` represents
        the sum of deltas, or the sum of the `reward` returned by `reward()`
        defined above. Think of it as the "true reward at this moment." I
        follow this convention for custom environments.

        For multi-step tasks such as `cloth-cover-item`, one must need to
        accomplish all stages correctly. HOWEVER, DO NOT use a condition here
        that says `done = (self.task_sage == X)` as that will prevent us from
        getting extra reward information, since in `self.reward()`, we first
        check if `done()` is True, then do the rewards. If it's a task-based
        done, then we immediately exit but don't compute the final reward
        statistics (undesirable).

        (08 Sept 2020): I added self.exit_gracefully, which will help us
        quickly exit if the demo has failed, but this is not handled here.

        (14 Sept 2020) We want self.done to return True if we should consider
        this episode a 'success'. If not, then if we return done=True with
        the understanding that it's a failure, we need to filter this episode
        elsewhere. So far the only tasks where this extra consideration has
        to be applied are bag-items-easy and bag-items-hard. See `main.py`
        and the `ignore_this_demo()` method.
        """
        zone_done, goal_done, cable_done, cov_done, bag_done = \
                False, False, False, False, False

        if self.metric == 'zone':
            zone_done = self.total_rewards == 1
        elif self.metric == 'cable-target':
            cable_done = self.total_rewards == 1
        elif self.metric == 'cable-ring':
            fraction = self.total_rewards / self.circle_area
            if fraction >= self.area_thresh:
                print(f'Frac: {fraction:0.4f} exceeds: {self.area_thresh:0.4f}')
                cable_done = True
        elif self.metric == 'cloth-coverage':
            if self.total_rewards >= self.coverage_threshold:
                cov_done = True
        elif self.metric == 'cloth-cover-item':
            cov_done = self.total_rewards == 1
        elif self.metric == 'bag-alone-open':
            fraction = self.total_rewards / self.circle_area
            if fraction >= self.area_thresh:
                print(f'Frac: {fraction:0.4f} exceeds: {self.area_thresh:0.4f}')
                bag_done = True
        elif self.metric == 'bag-items':
            bag_done = self.total_rewards > 0
        elif self.metric == 'bag-color-goal':
            bag_done = self.total_rewards >= 0.95

        # For envs with self.metric == 'pose'.
        if hasattr(self, 'goal'):
            goal_done = len(self.goal['steps']) == 0
        return zone_done or cable_done or cov_done or bag_done or goal_done

    #-------------------------------------------------------------------------
    # Environment Helper Functions
    #-------------------------------------------------------------------------

    def fill_template(self, template, replace):
        """Read template file and replace string keys.

        The replace[field] needs to be a tuple. Use (item,) for a single
        item, but you can combine keys with NAMEX where X starts from 0.
        """
        filepath = os.path.dirname(os.path.abspath(__file__))
        template = os.path.join(filepath, '..', template)
        with open(template, 'r') as file:
            fdata = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
        alphabet = string.ascii_lowercase + string.digits
        rname = ''.join(random.choices(alphabet, k=16))
        fname = f'{template}.{rname}'
        with open(fname, 'w') as file:
            file.write(fdata)
        return fname

    def random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get random box size."""
        size = np.random.rand(3)
        size[0] = size[0] * (max_x - min_x) + min_x
        size[1] = size[1] * (max_y - min_y) + min_y
        size[2] = size[2] * (max_z - min_z) + min_z
        return tuple(size)

    def get_object_masks(self, env):
        """Get RGB-D orthographic heightmaps and segmentation masks."""
        # TODO: speed this up with direct orthographic projection.

        # Capture RGB-D images and segmentation masks.
        color, depth = [], []
        for config in self.camera_config:
            color_t, depth_t, segm_t = env.render(config)
            color_t = np.concatenate((color_t, segm_t[..., None]), axis=2)
            color.append(color_t)
            depth.append(depth_t)

        # Reconstruct orthographic heightmaps with segmentation masks.
        heightmaps, colormaps = utils.reconstruct_heightmaps(
            color, depth, self.camera_config, self.bounds, self.pixel_size)
        masks = [colormap[..., 3:].squeeze() for colormap in colormaps]
        colormaps = np.array(colormaps)[..., :3]
        heightmaps = np.array(heightmaps)
        object_masks = np.array(masks)

        # Fuse heightmaps from different views.
        valid = np.sum(colormaps, axis=3) > 0
        repeat = np.sum(valid, axis=0)
        repeat[repeat == 0] = 1
        colormap = np.sum(colormaps, axis=0) / repeat[..., None]
        colormap = np.uint8(np.round(colormap))
        heightmap = np.max(heightmaps, axis=0)
        object_mask = np.max(object_masks, axis=0)

        return colormap, heightmap, object_mask

    def random_pose(self, env, object_size, hard_code=False):
        """Get random collision-free pose in workspace bounds for object.

        For tasks like sweeping to a zone target, generates a target zone at
        random, then later generates items to be swept in it. The second step
        requires free space to avoid being within the zone.

        The `mask` defines a distribution over pixels in the 320x160 top-down
        image, where 0=black means don't sample those pixels. For the
        remaining pixels (1=white), sample uniformly from those. It should
        correctly detect if we have a bag [but may require a low enough z
        coord]. If you increase `object_size`, then that reduces the
        available sampling area.

        Args:
            object_size: a tuple (x,y,z) used to determine erode size.
            hard_code: dummy argument added by me to debug positions.

        Returns:
            (position,rotation): used to convert from base / root link (I
                think) to where the object (or zone) should be located. The
                position must be sampled from the workspace, which is why
                environments with target zones have the zones sampled on the
                workstation. The rotation is only applied on the z-axis since
                we assume objects start on the same level surface (i.e.,
                workstation) and it's a simple uniform sample.
        """
        plane_id = 1
        max_size = np.sqrt(object_size[0]**2 + object_size[1]**2)
        erode_size = int(np.round(max_size / self.pixel_size))
        colormap, heightmap, object_mask = self.get_object_masks(env)

        # Sample freespace regions in workspace.
        mask = np.uint8(object_mask == plane_id)
        mask[0, :], mask[:, 0], mask[-1, :], mask[:, -1] = 0, 0, 0, 0
        mask = cv2.erode(mask, np.ones((erode_size, erode_size), np.uint8))
        if np.sum(mask) == 0:
            print('Warning! Sum of mask is zero in random_pose().')
            return
        pixel = utils.sample_distribution(np.float32(mask))
        position = utils.pixel_to_position(
            pixel, heightmap, self.bounds, self.pixel_size)
        position = (position[0], position[1], object_size[2] / 2)
        rtheta = np.random.rand() * 2 * np.pi
        rotation = p.getQuaternionFromEuler((0, 0, rtheta))

        if hard_code:
            # Daniel: to debug some hard-coded positions / quaternions.
            position = (0.0, 0.0, 0.0)
            rotation = p.getQuaternionFromEuler((0,0,0))
        return position, rotation

    def get_object_points(self, object_id):
        object_shape = p.getVisualShapeData(object_id)
        object_dim = object_shape[0][3]
        xv, yv, zv = np.meshgrid(
            np.arange(-object_dim[0] / 2, object_dim[0] / 2, 0.02),
            np.arange(-object_dim[1] / 2, object_dim[1] / 2, 0.02),
            np.arange(-object_dim[2] / 2, object_dim[2] / 2, 0.02),
            sparse=False, indexing='xy')
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

    def color_random_brown(self, object_id):
        shade = np.random.rand() + 0.5
        color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
        p.changeVisualShape(object_id, -1, rgbaColor=color)

    #-------------------------------------------------------------------------
    # Transformation Helper Functions
    #-------------------------------------------------------------------------

    def invert(self, pose):
        return p.invertTransform(pose[0], pose[1])

    def multiply(self, pose0, pose1):
        return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])

    def apply(self, pose, position):
        """Daniel: apply a rigid body transformation on `position` by `pose`.

        The `pose` has the rotation matrix and translation vector. Apply the
        rotation matrix on `position` and then translate. Example: used in
        cables env to sequentially translate each bead in the cable.

        Returns a *position*, not a pose (no orientation).
        """
        position = np.float32(position)
        position_shape = position.shape
        position = np.float32(position).reshape(3, -1)
        rotation = np.float32(p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
        translation = np.float32(pose[0]).reshape(3, 1)
        position = rotation @ position + translation
        return tuple(position.reshape(position_shape))

    #-------------------------------------------------------------------------
    # Image debugging.
    #-------------------------------------------------------------------------

    def save_images(self, colormap, heightmap, object_mask, pick_mask=None):
        os.makedirs('tmp', exist_ok=True)
        nb = len([x for x in os.listdir('tmp') if 'colormap' in x and '.png' in x])
        cv2.imwrite(f'tmp/{str(nb).zfill(3)}_colormap.png',
                    cv2.cvtColor((colormap).astype(np.uint8), cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'tmp/{str(nb).zfill(3)}_heightmap.png',
                    (heightmap / np.max(heightmap) * 255).astype(np.uint8))
        cv2.imwrite(f'tmp/{str(nb).zfill(3)}_object_mask.png',
                    (object_mask / np.max(object_mask) * 255).astype(np.uint8))