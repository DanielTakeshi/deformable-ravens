#!/usr/bin/env python
"""
Daniel's set of cloth-related robotics environments.

Interpreting self.zone_pose[0] = (x,y,z) where x and y are the VERTICAL and
HORIZONTAL (resp.) ranges in the diagram, in METERS:

  -0.5 .... +0.5
  ------------  0.3
  |          |   :
  |          |   :
  |          |   :
  ------------  0.7

The self.zone_pose corresponds to the colored yellow lines in the GUI. The
(0,0,0) location corresponds to the base of the robot.

NOTE: basePosition for the cloth is the CENTER of the cloth.

Some cloth hyper-parameters:
    Daniel: started w/mass=1.0 and elastic/damping stiffness of 40 and 0.1.
    Xuchen: if reducing mass of cloth, reduce springElasticStiffness.
    Xuchen: recommends springDampingAllDirections=0 for realistic damping.
See my internal Google Doc for additional information. It's ABSOLUTELY
CRITICAL to tune these, because it's hard to get realistic cloth. Also, note
that NxN cloth means the edge length is "cloth length" divided by N-1, not N.

For cloth vertices, we can get ground truth info. With 10x10 (therefore,
there are 9x9=81 actual _squares_ in the cloth mesh), indices are:

  90  -->  99
  80  -->  89
  ..  ...  ..
  10  -->  19
   0  -->   9

For 5x5 (so 4x4=16 squares) the indices are:

  20  -->  24
  15  -->  19
  10  -->  14
   5  -->   9
   0  -->   4

For a corner-pulling demonstrator, I like to grip 'one corner inwards'
instead of using the *actual* corners, except for the 5x5 case which might be
too coarse-grained. ACTUALLY ... we could just 'undershoot' if we want, but I
think it's easiest to do it this way.
"""
import os
import cv2
import time
import pkg_resources
import numpy as np
import pybullet as p
from ravens.tasks import Task
from ravens import utils as U


class ClothEnv(Task):
    """Superclass for cloth environments.

    Reward: cloth-flat-easy and cloth-pick-place use coverage. See comments
    in task.py, it's slightly roundabout. Demonstrator: corner-pulling
    demonstrator who uses access to underlying state information.

    In reset(), use the `replace` dict to adjust size of zone, and `scale`
    for the size of cloth. Note: the zone.obj ranges from (-10,10) whereas my
    cloth files from Blender scale from (-1,1), hence the cloth scale needs
    to be 10x *larger* than the zone scale. If this convention changes with
    any of the files, adjust accordingly. That will change how the zone looks
    in the images, and its corner indices.

    We also use `zone_size` to determine the object size for sampling. (And,
    in other environments where we must push items to a target, `zone_size`
    determines boundaries.)

    If I want at most N actions per episode, set max_steps=N+1 because the
    first has no primitive for whatever reason.
    """

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.primitive = 'pick_place'
        self.max_steps = 11
        self._IDs = {}
        self._debug = True
        self._settle_secs = 0

        # Gripping parameters. I think _def_threshold = 0.020 is too low.
        self._def_threshold = 0.025
        self._def_nb_anchors = 1

        # See scaling comments above.
        self._zone_scale = 0.01
        self._cloth_scale = 0.10
        self._cloth_length = (2.0 * self._cloth_scale)
        self._zone_length = (20.0 * self._zone_scale)
        self.zone_size = (20.0 * self._zone_scale, 20.0 * self._zone_scale, 0)
        self._cloth_size = (self._cloth_length, self._cloth_length, 0.01)
        assert self._cloth_scale == self._zone_scale * 10, self._cloth_scale

        # Cloth resolution and corners (should be clockwise).
        self.n_cuts = 10
        if self.n_cuts == 5:
            self.corner_indices = [0, 20, 24, 4]  # actual corners
        elif self.n_cuts == 10:
            self.corner_indices = [11, 81, 88, 18]  # one corner inwards
        else:
            raise NotImplementedError(self.n_cuts)
        self._f_cloth = 'assets/cloth/bl_cloth_{}_cuts.obj'.format(
                str(self.n_cuts).zfill(2))

        # Other cloth parameters.
        self._mass = 0.5
        self._edge_length = (2.0 * self._cloth_scale) / (self.n_cuts - 1)
        self._collisionMargin = self._edge_length / 5.0

        # IoU/coverage rewards (both w/zone or goal images). Pixels w/255 are targets.
        self.target_hull_bool = None
        self.zone_ID = -1

    def get_target_zone_corners(self):
        """Determine corners of target zone.

        Assumes we follow this structure in some order:
          c2 --- c3
          |       |
          |       |
          c1 --- c4
        The actual xyz positions depend on the sampled `self.zone_pose`.

        We use this for (among other things) the target points for the
        ClothPickPlace demonstrator. Make sure the clockwise ordering is
        consistent with labels, for the reward and demonstrator. NOTE: the
        zone stays clockwise in terms of our c1 -> c2 -> c3 -> c4 ordering
        (from a top down view) but the CLOTH may have been flipped. For now I
        assume that can be handled in the action by considering the possible
        counter-clockwise map.
        """
        EL2 = self._zone_length / 2
        self.c1_position = self.apply(self.zone_pose, (-EL2, -EL2, 0))
        self.c2_position = self.apply(self.zone_pose, (-EL2,  EL2, 0))
        self.c3_position = self.apply(self.zone_pose, ( EL2,  EL2, 0))
        self.c4_position = self.apply(self.zone_pose, ( EL2, -EL2, 0))
        self._corner_targets_xy = np.array([
            self.c1_position[:2],
            self.c2_position[:2],
            self.c3_position[:2],
            self.c4_position[:2],
        ])

    def get_masks_target(self, env, zone_ID):
        """For getting a mask of the cloth's target zone.

        We can then use this as a target, and compute a simple, pixel-based
        IoU metric for the reward. This is likely more robust than the prior
        reward computation where I check the cloth vertices and see if they
        are close to a grid point. Does not apply for cloth tasks w/goal
        images, but we use a similar idea for those tasks.

        https://github.com/DanielTakeshi/pybullet-def-envs/pull/12
        https://github.com/DanielTakeshi/pybullet-def-envs/pull/13

        NOTE: to make things easier for goal-conditioned envs, we should also
        save the `target_hull_bool` with each episode -- put in `last_info`.
        """
        _, _, object_mask = self.get_object_masks(env)

        # Check object_mask for all pixels corresponding to the zone ID.
        mask = np.float32(object_mask == zone_ID)
        mask = np.uint8(mask * 255)

        # Find contours of the `mask` image, combine all to get shape (N,1,2).
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_list = [ np.concatenate([c for c in contours]) ]

        # Find the convex hull object for that combined contour.
        hull_list = [cv2.convexHull(c) for c in contours_list]

        # Make an RGB image, then draw the filled-in area of all items in `hull_list`.
        hull = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(hull, hull_list, -1, (255,255,255), thickness=-1)

        # Assign to `target_hull_bool` so we continually check it each time step.
        target_hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)
        self.target_hull_bool = np.array(target_hull, dtype=bool)

    def compute_pixel_IoU_coverage(self):
        """Computes IoU and coverage based on pixels.

        Use: `self.target_hull` and `self.current_hull`. For the former:
        values of 255 refer to the area contained within the zone (and
        includes the zone itself, FWIW). For the latter: segment to detect
        the workspace OR the zone ID. Then use numpy and/or operators:

        https://stackoverflow.com/questions/49338166/python-intersection-over-union
        https://stackoverflow.com/questions/14679595/vs-and-vs

        NOTE I: assumes cloth can be segmented by detecting the workspace and
        zone lines, and that any pixel OTHER than those belongs to cloth.

        NOTE II: IoU and coverage are computed in the same way, except that
        the former divides by the union, the latter divides by just the goal.
        """
        _, _, object_mask = self.get_object_masks(self.env)

        # Check object_mask for all pixels OTHER than the cloth.
        IDs = [1, self.zone_ID]  # 1 = workspace_ID
        mask = np.isin(object_mask, test_elements=IDs)

        # Flip items so that 1 = cloth (then 255 when scaled).
        idx_0s = (mask == 0)
        idx_1s = (mask == 1)
        mask[idx_0s] = 1
        mask[idx_1s] = 0
        cloth_mask_bool = np.array(mask, dtype=bool)

        # Compute pixel-wise IoU and coverage, using two bool dtype arrays.
        overlap        = self.target_hull_bool & cloth_mask_bool  # Logical AND
        union          = self.target_hull_bool | cloth_mask_bool  # Logical OR
        overlap_count  = np.count_nonzero(overlap)
        union_count    = np.count_nonzero(union)
        goal_count     = np.count_nonzero(self.target_hull_bool)
        pixel_IoU      = overlap_count / float(union_count)
        pixel_coverage = overlap_count / float(goal_count)
        return (pixel_IoU, pixel_coverage)

    def is_item_covered(self):
        """For cloth-cover, if it's covered, it should NOT be in the mask."""
        _, _, object_mask = self.get_object_masks(self.env)
        assert len(self.block_IDs) == 1, self.block_IDs
        block = self.block_IDs[0]
        return 1 - float(block in object_mask)

    def add_zone(self, env, zone_pose=None):
        """Adds a square target (green) zone.

        To handle goal-conditioned cloth flattening, we save `zone_pose` and
        provide it as input, to avoid re-sampling. This means starting cloth
        states are sampled at a correct distance, and that the same IoU
        metric can be used as the reward.
        """
        zone_template = 'assets/zone/zone-template.urdf'
        replace = {'LENGTH': (self._zone_scale, self._zone_scale)}
        zone_urdf = self.fill_template(zone_template, replace)

        # Pre-assign zone pose _only_ if loading goal-conditioned policies.
        if zone_pose is not None:
            self.zone_pose = zone_pose
        else:
            self.zone_pose = self.random_pose(env, self.zone_size)

        # For tracking IDs and consistency with existing ravens code.
        zone_id = env.add_object(zone_urdf, self.zone_pose, fixed=True)
        self._IDs[zone_id] = 'zone'
        self.get_masks_target(env, zone_id)
        os.remove(zone_urdf)

        # To reference it later for IoUs/coverage, or to remove if needed.
        self.zone_ID = zone_id

    def add_cloth(self, env, base_pos, base_orn):
        """Adding a cloth from an .obj file."""
        cloth_id = p.loadSoftBody(
                fileName=self._f_cloth,
                basePosition=base_pos,
                baseOrientation=base_orn,
                collisionMargin=self._collisionMargin,
                scale=self._cloth_scale,
                mass=self._mass,
                useNeoHookean=0,
                useBendingSprings=1,
                useMassSpring=1,
                springElasticStiffness=40,
                springDampingStiffness=0.1,
                springDampingAllDirections=0,
                useSelfCollision=1,
                frictionCoeff=1.0,
                useFaceContact=1,)

        # Only if using more recent PyBullet versions.
        p_version = pkg_resources.get_distribution('pybullet').version
        if p_version == '3.0.4':
            color = U.COLORS['yellow'] + [1]
            p.changeVisualShape(cloth_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                                rgbaColor=color)

        # For tracking IDs and consistency with existing ravens code.
        self._IDs[cloth_id] = 'cloth'
        self.object_points[cloth_id] = np.float32((0, 0, 0)).reshape(3, 1)
        env.objects.append(cloth_id)

        # To help environment pick-place method track all deformables.
        self.def_IDs.append(cloth_id)

        # Sanity checks.
        nb_vertices, _ = p.getMeshData(cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        assert nb_vertices == self.n_cuts * self.n_cuts
        return cloth_id

    def _sample_cloth_orientation(self):
        """Sample the bag (and let it drop) to get interesting starting states."""
        orn = [self._base_orn[0] + np.random.normal(loc=0.0, scale=self._scalex),
               self._base_orn[1] + np.random.normal(loc=0.0, scale=self._scaley),
               self._base_orn[2] + np.random.normal(loc=0.0, scale=self._scalez),]
        return p.getQuaternionFromEuler(orn)

    def get_ID_tracker(self):
        return self._IDs

    @property
    def coverage_threshold(self):
        return self._coverage_thresh

    @property
    def corner_targets_xy(self):
        return self._corner_targets_xy

    @property
    def def_threshold(self):
        return self._def_threshold

    @property
    def def_nb_anchors(self):
        return self._def_nb_anchors

    def debug(self):
        np.set_printoptions(suppress=True, edgeitems=10, linewidth=200)
        _, vert_pos_l = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        _, _, c3, c4 = self.c1, self.c2, self.c3, self.c4
        print('\nInside {} reset()'.format(self._name))
        print('\tcloth grid:      {} x {}'.format(self.n_cuts, self.n_cuts))
        print('\tcloth scale:     {:.3f}'.format(self._cloth_scale))
        print('\tedge_length:     {:.3f} meters'.format(self._edge_length))
        print('\tcollisionMargin: {:.3f} meters ({:.1f}) mm'.format(
                self._collisionMargin, self._collisionMargin*1000))
        print('\tcloth mass:      {:.3f} kg'.format(self._mass))
        print('zone_pose:    {}'.format(U.round_pose(self.zone_pose)))
        print('zone c1 pos:  {}'.format(U.round_pos(self.c1_position)))
        print('zone c2 pos:  {}'.format(U.round_pos(self.c2_position)))
        print('zone c3 pos:  {}'.format(U.round_pos(self.c3_position)))
        print('zone c4 pos:  {}'.format(U.round_pos(self.c4_position)))
        print('cloth c1 pos: {}'.format(U.round_pos(vert_pos_l[self.corner_indices[0]])))
        print('cloth c2 pos: {}'.format(U.round_pos(vert_pos_l[self.corner_indices[1]])))
        print('cloth c3 pos: {}'.format(U.round_pos(vert_pos_l[self.corner_indices[2]])))
        print('cloth c4 pos: {}'.format(U.round_pos(vert_pos_l[self.corner_indices[3]])))
        print(f'alpha_l: {self.alpha_l}')
        print(f'c4 {c4}, c3 {c3}, edge_length {self._edge_length:0.3f}, vec_offset {self.vec_offset}')
        print('Waiting {} secs for cloth to settle ...\n'.format(self._settle_secs))


class ClothFlat(ClothEnv):
    """
    We start with a flat cloth that starts relatively close to a visible target
    zone, and usually overlaps (but will not trigger coverage threshold at the
    start). Sample the zone and cloth centers, then determine the actual cloth
    center by taking a point along the vector of the two positions.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 11
        self.metric = 'cloth-coverage'
        self._name = 'cloth-flat'
        self._debug = False
        self._coverage_thresh = 0.85

        # Env reference so we can call Task.get_object_masks(env)
        self.env = None

        # Cloth sampling. Max zone distance (cloth-to-zone) heavily tuned.
        self._scalex = 0.0
        self._scaley = 0.0
        self._scalez = 0.5
        self._base_orn = [np.pi / 2.0, 0, 0]
        self._drop_height = 0.01
        self._max_zone_dist = 0.34

        # Action parameters.
        self.primitive_params = {
            1: {'speed': 0.002,
                'delta_z': -0.0010,
                'postpick_z': 0.05,
                'preplace_z': 0.05,
                'pause_place': 0.0,
            }
        }
        self.task_stage = 1

    def reset(self, env, zone_pose=None):
        self.total_rewards = 0
        self.object_points = {}
        self.t = 0
        self.task_stage = 1
        self.def_IDs = []
        self.env = env

        # Add square target zone, determine corners, handle reward function.
        self.add_zone(env, zone_pose=zone_pose)
        self.get_target_zone_corners()

        # Used to sample a pose that is sufficiently close to the zone center.
        def cloth_to_zone(bpos):
            p1 = np.float32(bpos)
            p2 = np.float32(self.zone_pose[0])
            return np.linalg.norm(p1 - p2)

        # Sample a flat cloth position somewhere on the workspace.
        bpos, _ = self.random_pose(env, self._cloth_size)
        while cloth_to_zone(bpos) > self._max_zone_dist:
            if self._debug:
                print(f'NOTE: {cloth_to_zone(bpos):0.3f} > {self._max_zone_dist:0.3f} '
                        '= max zone dist, re-sampling...')
            bpos, _ = self.random_pose(env, self._cloth_size)

        # Make cloth closer to the zone, sample orientation, and create it.
        alpha = 0.6
        bpos_x = (bpos[0] * alpha) + (self.zone_pose[0][0] * (1 - alpha))
        bpos_y = (bpos[1] * alpha) + (self.zone_pose[0][1] * (1 - alpha))
        self.base_pos = [bpos_x, bpos_y, self._drop_height]
        self.base_orn = self._sample_cloth_orientation()
        self.cloth_id = self.add_cloth(env, self.base_pos, self.base_orn)

        if self._debug:
            self.debug()
        env.start()
        time.sleep(self._settle_secs)
        env.pause()


class ClothFlatNoTarget(ClothFlat):
    """Like ClothFlat, except no visible targets."""

    def __init__(self):
        super().__init__()
        self._name = 'cloth-flat-notarget'
        self._debug = False

    def reset(self, env, last_info=None):
        """Reset to start an episode.

        Call the superclass to generate as usual, and then remove the zone
        here. Requires care in `environment.py` to avoid iterating over
        invalid IDs, and we need the zone in the superclass for many reasons.

        If loading goal images, we cannot just override self.target_hull_bool
        because that means the cloth isn't sampled the same way as if the
        target was visible. We need to first decide on the pose, THEN sample
        the cloth. Easiset solution: load the sampled `zone_pose`, then pass
        that into the reset() call so that we don't re-sample `zone_pose`.
        Everything is reconsructed from there.
        """
        zone_pose = None
        if last_info is not None:
            zone_pose = last_info['sampled_zone_pose']
        super().reset(env, zone_pose=zone_pose)
        p.removeBody(self.zone_ID)


class ClothCover(ClothEnv):
    """Now consider cloth and items, where the cloth should cover it.

    There is no 'target zone' here. We just want to cover items. For now,
    limit to two actions.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 2 + 1
        self.metric = 'cloth-cover-item'
        self._name = 'cloth-cover'
        self._debug = False

        # Env reference so we can call Task.get_object_masks(env)
        self.env = None

        # Cloth sampling.
        self._scalex = 0.0
        self._scaley = 0.0
        self._scalez = 0.5
        self._base_orn = [np.pi / 2.0, 0, 0]
        self._drop_height = 0.01

        # Action parameters. Make postpick_z a bit higher vs cloth-flat.
        self.primitive_params = {
            1: {'speed': 0.003,
                'delta_z': -0.0005,
                'postpick_z': 0.10,
                'preplace_z': 0.10,
                'pause_place': 0.0,
            },
            2: {'speed': 0.003,
                'delta_z': -0.0005,
                'postpick_z': 0.10,
                'preplace_z': 0.10,
                'pause_place': 0.0,
            },
        }
        self.task_stage = 1

        # Extra non-cloth items.
        self._nb_blocks = 1
        self.block_IDs = []

    def reset(self, env):
        self.total_rewards = 0
        self.object_points = {}
        self.t = 0
        self.task_stage = 1
        self.def_IDs = []
        self.block_IDs = []
        self.env = env

        # Add blocks (following sorting environment).
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'assets/stacking/block.urdf'
        for _ in range(self._nb_blocks):
            block_pose = self.random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            self.object_points[block_id] = np.float32((0, 0, 0)).reshape(3, 1)
            self._IDs[block_id] = 'block'
            self.block_IDs.append(block_id)

        # Sample a flat cloth arbitrarily on the workspace.
        bpos, _ = self.random_pose(env, self._cloth_size)
        self.base_pos = [bpos[0], bpos[1], self._drop_height]
        self.base_orn = self._sample_cloth_orientation()
        self.cloth_id = self.add_cloth(env, self.base_pos, self.base_orn)

        if self._debug:
            self.debug()
        env.start()
        time.sleep(self._settle_secs)
        env.pause()
