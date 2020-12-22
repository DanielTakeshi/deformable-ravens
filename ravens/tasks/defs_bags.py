#!/usr/bin/env python
"""
Daniel's set of bag-related robotics environments.

Interpreting self.zone_pose[0] = (x,y,z) where x and y are the vertical and
horizontal (resp.) ranges in the diagram:

  -0.5 .... +0.5
  ------------  0.3
  |          |   :
  |          |   :
  |          |   :
  ------------  0.7

The self.zone_pose corresponds to the colored yellow lines in the GUI. The
(0,0,0) location corresponds to the base of the robot.

Bag type and radius at the start:
- bag 1, radius at very start: 0.1000 (whew!)
- bag 2, radius at very start: 0.0981
- bag 3, radius at very start: 0.0924
- bag 4, radius at very start: 0.0831
- bag 5, radius at very start: 0.0707

See the classes for detailed documentation on actions, rewards, etc.
"""
import os
import cv2
import time
import pkg_resources
import numpy as np
import pybullet as p
from ravens.tasks import Task
from ravens import utils as U

BAGS_TO_FILES = {
    1: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.1_numV_257.obj',
    2: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.3_numV_289.obj',
    3: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.4_numV_321.obj',
    4: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.6_numV_353.obj',
    5: 'assets/bags/bl_sphere_bag_rad_1.0_zthresh_0.8_numV_385.obj',
}

# An identity pose we can use to gracefully handle failure cases.
IDENTITY = {'pose0': ((0.3,0,0.3), (0,0,0,1)), 'pose1': ((0.3,0,0.3), (0,0,0,1))} #TODO(daniel) remove
BEAD_THRESH = 0.33 #TODO(daniel) make cleaner


class BagEnv(Task):
    """Superclass to reduce code duplication.

    Gripping parameter: the threshold should probably be lower compared to
    the cloth tasks, since for our bags, we generally want to grip the beads,
    instead of the bag vertices.
    """

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.primitive = 'pick_place'
        self.max_steps = 11
        self._IDs = {}
        self._debug = True
        self._settle_secs = 1

        # Gripping parameters. See docs above.
        self._def_threshold = 0.020
        self._def_nb_anchors = 1

        # Scale the bag / zone. The zone.obj ranges from (-20,20).
        self._zone_scale = 0.0130
        self._bag_scale = 0.10
        self._zone_length = (20. * self._zone_scale)
        self.zone_size = (20. * self._zone_scale, 20. * self._zone_scale,  0.0)
        self._bag_size = (  1. * self._bag_scale,   1. * self._bag_scale, 0.01)

        # Bag type (or resolution?) and parameters.
        self._bag = 4
        self._mass = 1.0
        self._scale = 0.25
        self._collisionMargin = 0.003
        self._base_orn = [np.pi / 2.0, 0.0, 0.0]
        self._f_bag = BAGS_TO_FILES[self._bag]
        self._drop_height = 0.15

    def get_target_zone_corners(self):
        """Determine corners of target zone.

        Assumes we follow this structure in some order:
          c2 --- c3
          |       |
          |       |
          c1 --- c4
        The actual xyz positions depend on the sampled `self.zone_pose`.
        """
        EL2 = self._zone_length / 2
        self.c1_position = self.apply(self.zone_pose, (-EL2, -EL2, 0))
        self.c2_position = self.apply(self.zone_pose, (-EL2,  EL2, 0))
        self.c3_position = self.apply(self.zone_pose, ( EL2,  EL2, 0))
        self.c4_position = self.apply(self.zone_pose, ( EL2, -EL2, 0))

    def add_zone(self, env):
        """Adds a square target (green) zone."""
        zone_template = 'assets/zone/zone-template.urdf'
        replace = {'LENGTH': (self._zone_scale, self._zone_scale)}
        zone_urdf = self.fill_template(zone_template, replace)
        self.zone_pose = self.random_pose(env, self.zone_size)

        # For tracking IDs and consistency with existing ravens code.
        zone_id = env.add_object(zone_urdf, self.zone_pose, fixed=True)
        os.remove(zone_urdf)
        self._IDs[zone_id] = 'zone'

        # As in `defs_cloth.py`, ro reference it later, e.g., for removal.
        self.zone_ID = zone_id

        return zone_id

    def add_cable_ring(self, env, bag_id=None):
        """Make the cable beads coincide with the vertices of the top ring.

        This should lead to better physics and will make it easy for an
        algorithm to see the bag's top ring. Please see the cable-ring env
        for details, or `scratch/cable_ring_MWE.py`. Notable differences
        (or similarities) between this and `dan_cables.py`:

        (1) We don't need to discretize rotations and manually compute bead
        positions, because the previously created bag 'creates' it for us.

        (2) Beads have anchors with vertices, in addition to constraints with
        adjacent beads.

        (3) Still use `self.cable_bead_IDs` as we want that for the reward.
        """
        num_parts = len(self._top_ring_idxs)
        radius = 0.005
        color = U.COLORS['blue'] + [1]
        beads = []
        bead_positions_l = []
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius]*3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius*1.5)

        # All tasks OTHER than bag-color-goal use self.bag_id. So if we are not
        # passing it in as an argument, we better have defined self.bag_id to use.
        if bag_id is None:
            bag_id = self.bag_id

        # Fortunately `verts_l` coincides with `self._top_ring_idxs`.
        _, verts_l = p.getMeshData(bag_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)

        # Iterate through parts and create constraints as needed.
        for i in range(num_parts):
            bag_vidx = self._top_ring_idxs[i]
            bead_position = np.float32(verts_l[bag_vidx])
            part_id = p.createMultiBody(0.01, part_shape, part_visual,
                    basePosition=bead_position)
            p.changeVisualShape(part_id, -1, rgbaColor=color)

            if i > 0:
                parent_frame = bead_position - bead_positions_l[-1]
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=beads[-1],
                        parentLinkIndex=-1,
                        childBodyUniqueId=part_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)

            # Make a constraint with i=0. Careful with `parent_frame`!
            if i == num_parts - 1:
                parent_frame = bead_positions_l[0] - bead_position
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=part_id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=beads[0],
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)

            # Create constraint between a bead and certain bag vertices.
            _ = p.createSoftBodyAnchor(
                    softBodyBodyUniqueId=bag_id,
                    nodeIndex=bag_vidx,
                    bodyUniqueId=part_id,
                    linkIndex=-1,)

            # Track beads.
            beads.append(part_id)
            bead_positions_l.append(bead_position)

            # The usual for tracking IDs. Four things to add.
            self.cable_bead_IDs.append(part_id)
            self._IDs[part_id] = f'cable_part_{str(part_id).zfill(2)}'
            env.objects.append(part_id)
            self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

    def add_bag(self, env, base_pos, base_orn, bag_color='yellow'):
        """Adding a bag from an .obj file."""
        bag_id = p.loadSoftBody(
                fileName=self._f_bag,
                basePosition=base_pos,
                baseOrientation=base_orn,
                collisionMargin=self._collisionMargin,
                scale=self._bag_scale,
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
            color = U.COLORS[bag_color] + [1]
            p.changeVisualShape(bag_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                                rgbaColor=color)
        else:
            raise ValueError(p_version)

        # For tracking IDs and consistency with existing ravens code.
        self._IDs[bag_id] = 'bag'
        self.object_points[bag_id] = np.float32((0, 0, 0)).reshape(3, 1)
        env.objects.append(bag_id)

        # To help environment pick-place method track all deformables.
        self.def_IDs.append(bag_id)

        return bag_id

    def _sample_bag_orientation(self):
        """Sample the bag (and let it drop) to get interesting starting states."""
        orn = [self._base_orn[0] + np.random.normal(loc=0.0, scale=self._scale),
               self._base_orn[1] + np.random.normal(loc=0.0, scale=self._scale),
               self._base_orn[2] + np.random.normal(loc=0.0, scale=self._scale),]
        return p.getQuaternionFromEuler(orn)

    def get_ID_tracker(self):
        return self._IDs

    @property
    def circle_area(self):
        return self._circle_area

    @property
    def area_thresh(self):
        """Testing with bag-alone-open, similar to cable-ring, slightly lower?"""
        return 0.70

    @property
    def circle_target_positions(self):
        return self._target_positions

    @property
    def circle_target_center(self):
        return self._circle_center

    @property
    def top_ring_idxs(self):
        return self._top_ring_idxs

    @property
    def def_threshold(self):
        return self._def_threshold

    @property
    def def_nb_anchors(self):
        return self._def_nb_anchors

    def understand_bag_top_ring(self, env, base_pos):
        """By our circular bag design, there exists a top ring file.

        Reading it gives us several important pieces of information. We assign to:

            _top_ring_idxs: indices of the vertices (out of entire bag).
            _top_ring_posi: their starting xyz positions (BEFORE simulation
                or applying pose transformations). This way we can get the
                area of the circle. We can't take the rotated bag and map
                vertices to the xy plane, because any rotation will make the
                area artificially smaller.

        The .txt file saves in (x,y,z) order but the .obj files put z second.
        Make sure vertex indices are MONOTONICALLY INCREASING since I use
        that assumption to 'assign' vertex indices in order to targets.

        Input: base_pos, the center of the bag's sphere.
        """
        self._top_ring_f = (self._f_bag).replace('.obj', '_top_ring.txt')
        self._top_ring_f = os.path.join('ravens', self._top_ring_f)
        self._top_ring_idxs = [] # is this the same as p.getMeshData?
        self._top_ring_posi = [] # for raw, non-scaled bag
        with open(self._top_ring_f, 'r') as fh:
            for line in fh:
                ls = (line.rstrip()).split()
                vidx = int(ls[0])
                vx, vy, vz = float(ls[1]), float(ls[2]), float(ls[3])
                if len(self._top_ring_idxs) >= 1:
                    assert vidx > self._top_ring_idxs[-1], \
                            f'Wrong: {vidx} vs {self._top_ring_idxs}'
                self._top_ring_idxs.append(vidx)
                self._top_ring_posi.append((vx,vy,vz))

        # Next, define a target zone. This makes a bunch of plus signs in a
        # circular fashion from the xy projection of the ring.
        self._target_positions = []
        for item in self._top_ring_posi:
            sx, sy, _ = item
            sx = sx * self._bag_scale + base_pos[0]
            sy = sy * self._bag_scale + base_pos[1]
            self._target_positions.append( (sx,sy,0) )
            if self._targets_visible:
                square_pose = ((sx,sy,0.001), (0,0,0,1))
                square_template = 'assets/square/square-template-allsides-green.urdf'
                replace = {'DIM': (0.004,), 'HALF': (0.004 / 2,)}
                urdf = self.fill_template(square_template, replace)
                env.add_object(urdf, square_pose, fixed=True)
                os.remove(urdf)

        if self._name in ['bag-color-goal']:
            return

        # Fit a circle and print some statistics, can be used by demonstrator.
        # We should be careful to consider nonplanar cases, etc.
        xc, yc, rad, _ = U.fit_circle(self._top_ring_posi, self._bag_scale, debug=False)
        self._circle_area = np.pi * (rad ** 2)
        self._circle_center = (xc * self._bag_scale + base_pos[0],
                               yc * self._bag_scale + base_pos[1])

    def debug(self):
        np.set_printoptions(suppress=True, edgeitems=10, linewidth=200)
        print('\nInside {} reset()'.format(self._name))
        print('\tbag mass:        {:.3f} kg'.format(self._mass))
        print('\tbag scale:       {:.3f}'.format(self._bag_scale))
        print('\tcollisionMargin: {:.3f} meters ({:.1f}) mm'.format(
                self._collisionMargin, self._collisionMargin*1000))
        print(f'\tbag sampled base_pos: {self.base_pos}')
        print(f'\tbag sampled base_orn: {self.base_orn}')
        try:
            print('top ring verts: {}'.format(self._top_ring_idxs))
        except:
            pass
        print('Waiting {} secs for stuff to settle ...'.format(self._settle_secs))

    def _apply_small_force(self, num_iters, fx=10, fy=10, fz=8):
        """A small force to perturb the starting bag."""

        # First bag. Assume that 32 beads are in one bag.
        bead_idx = np.random.randint(len(self.cable_bead_IDs))
        bead_id = self.cable_bead_IDs[bead_idx]
        fx_1 = np.random.randint(low=-fx, high=fx + 1)
        fy_1 = np.random.randint(low=-fy, high=fy + 1)

        # Second bag if necessary.
        if len(self.cable_bead_IDs) > 32:
            assert len(self.cable_bead_IDs) == 64, len(self.cable_bead_IDs)
            if bead_idx < 32:
                bead_idx_2 = np.random.randint(32, len(self.cable_bead_IDs))
            else:
                bead_idx_2 = np.random.randint(0, 32)
            bead_id_2 = self.cable_bead_IDs[bead_idx_2]
            fx_2 = np.random.randint(low=-fx, high=fx + 1)
            fy_2 = np.random.randint(low=-fy, high=fy + 1)

        for _ in range(num_iters):
            p.applyExternalForce(bead_id, linkIndex=-1, forceObj=[fx_1, fy_1, fz],
                                 posObj=[0,0,0], flags=p.LINK_FRAME)
            if len(self.cable_bead_IDs) > 32:
                p.applyExternalForce(bead_id_2, linkIndex=-1, forceObj=[fx_2, fy_2, fz],
                                     posObj=[0,0,0], flags=p.LINK_FRAME)

            if self._debug:
                print(f'Perturbing {bead_id}: [{fx:0.2f}, {fy:0.2f}, {fz:0.2f}]')


class BagAloneOpen(BagEnv):
    """Just a single bag, and this time we have to OPEN the top ring.

    Reward: I am going to use the same reward as `cable-ring` given that they
    both share a similar ring of beads, and that increasing the convex hull
    will usually mean it's better. BUT ... huge caveat, we really should have
    a visibility check of some sort, because we could have the convex hull be
    entirely covered by the bag. TODO: how do we do that?

    Currently using max_steps=10+1 as a way to balance out speed versus
    getting a reasonable opening policy. The cable-ring environments have
    more complex starting rings.

    We create a fake zone and delete it, so that the bag color looks the same
    as compared to bag-items-{easy,hard}.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 8+1
        self.metric = 'bag-alone-open'
        self._name = 'bag-alone-open'
        self._targets_visible = False
        self._debug = False

        # Make the scale small as it's just to make the bag a certain color.
        self._zone_scale = 0.004

        # Higher means more forces applied to the bag.
        self.num_force_iters = 12

        # Parameters for pick_place primitive. Setting prepick_z to be <0.3
        # because it takes a while to check against gripping deformables.
        self.primitive_params = {
            1: {'speed': 0.003,
                'delta_z': -0.001,
                'prepick_z': 0.10,
                'postpick_z': 0.05,
                'preplace_z': 0.05,
                'pause_place': 0.5,
            },
        }
        self.task_stage = 1

    def reset(self, env):
        self.total_rewards = 0
        self.object_points = {}
        self.t = 0
        self.task_stage = 1
        self.cable_bead_IDs = []
        self.def_IDs = []

        # Add square target zone only to get the bag a certain color.
        self.add_zone(env)

        # Pose of the bag, sample mid-air to let it drop naturally.
        bpos, _ = self.random_pose(env, self._bag_size)
        self.base_pos = [bpos[0], bpos[1], self._drop_height]
        self.base_orn = self._sample_bag_orientation()

        # Add the bag, load info about top ring, and make a cable.
        self.bag_id = self.add_bag(env, self.base_pos, self.base_orn)
        self.understand_bag_top_ring(env, self.base_pos)
        self.add_cable_ring(env)

        # Env must begin before we can apply forces.
        if self._debug:
            self.debug()
        env.start()

        # Add a small force to perturb the bag.
        self._apply_small_force(num_iters=self.num_force_iters)

        # Remove the zone ID -- only had this to make the bag color the same.
        p.removeBody(self.zone_ID)

        time.sleep(self._settle_secs)
        env.pause()


class BagItemsEasy(BagEnv):
    """Like BagAlone except we add other stuff.

    Right now I'm trying to make the demonstrator follow one of three stages,
    where the stages are bag opening, item insertion, and bag moving. For a
    consistant API among other 'bag-items' environments, please put all items
    to be inserted in `self.item_IDs[]` and use `self.items_in_bag_IDs` to
    track those IDs which are already inserted (or at least, which the
    demonstrator thinks is inserted).
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 8+1
        self.metric = 'bag-items'
        self._name = 'bag-items-easy'
        self._targets_visible = False
        self._debug = False

        # Can make this smaller compared to bag-items-alone.
        self.num_force_iters = 8

        # Extra items, in addition to the bag.
        self._nb_items = 1

        # Env reference so we can call Task.get_object_masks(env)
        self.env = None

        # Parameters for pick_place primitive, which is task dependent.
        # stage 1: bag opening. [Copying params from bag-alone-open, except increasing speed]
        # stage 2: item insertion.
        # stage 3: bag pulling.
        self.primitive_params = {
            1: {'speed': 0.004,
                'delta_z': -0.001,
                'prepick_z': 0.10,
                'postpick_z': 0.05,
                'preplace_z': 0.05,
                'pause_place': 0.25,
            },
            2: {'speed': 0.010,
                'delta_z': -0.0005, # same as cloth-cover
                'prepick_z': 0.10,  # hopefully makes it faster
                'postpick_z': 0.30,
                'preplace_z': 0.30,
                'pause_place': 0.0,
            },
            3: {'speed': 0.002,  # Will this slow bag movement?
                'delta_z': -0.001,
                'prepick_z': 0.08,  # hopefully makes it faster
                'postpick_z': 0.40,
                'preplace_z': 0.40,
                'pause_place': 0.50,  # used to be 2, maybe make smaller?
            },
        }
        self.task_stage = 1

    def reset(self, env):
        self.total_rewards = 0
        self.object_points = {}
        self.t = 0
        self.task_stage = 1
        self.cable_bead_IDs = []
        self.def_IDs = []
        self.env = env
        self.exit_gracefully = False

        # New stuff versus bag-alone-open, to better track stats.
        self.item_IDs = []
        self.items_in_bag_IDs = []
        self.item_sizes = []

        # Add square target zone and determine its corners.
        self.add_zone(env)
        self.get_target_zone_corners()

        # Pose of the bag, sample mid-air to let it drop naturally.
        bpos, _ = self.random_pose(env, self._bag_size)
        self.base_pos = [bpos[0], bpos[1], self._drop_height]
        self.base_orn = self._sample_bag_orientation()

        # Add the bag, load info about top ring, and make a cable.
        self.bag_id = self.add_bag(env, self.base_pos, self.base_orn)
        self.understand_bag_top_ring(env, self.base_pos)
        self.add_cable_ring(env)

        # Add cube.
        item_size = (0.04, 0.04, 0.04)
        for _ in range(self._nb_items):
            item_pose = self.random_pose(env, item_size)
            item_id = self.add_cube(env, pose=item_pose, globalScaling=0.90)  # reduced from 1.0 for training.
            self.item_IDs.append(item_id)
            self.item_sizes.append(item_size)

        # Env must begin before we can apply forces.
        if self._debug:
            self.debug()
        env.start()

        # Add a small force to perturb the bag.
        self._apply_small_force(num_iters=self.num_force_iters)
        time.sleep(self._settle_secs)

        # Check that all added blocks are visible.
        colormap, heightmap, object_mask = self.get_object_masks(env)
        for ID in self.item_IDs:
            if ID not in object_mask:
                print(f'Warning, ID={ID} not in object_mask during reset(). Exit!')
                self.exit_gracefully = True
                self.save_images(colormap, heightmap, object_mask)
        env.pause()

    def add_cube(self, env, pose, globalScaling=1.0):
        """Andy's ravens/block's default size should be (0.04, 0.04, 0.04)."""
        cube_id = p.loadURDF(
                fileName='assets/block/block_for_anchors.urdf',
                basePosition=pose[0],
                baseOrientation=pose[1],
                globalScaling=globalScaling,
                useMaximalCoordinates=True)

        # For tracking IDs and consistency with existing ravens code.
        self._IDs[cube_id] = 'cube'
        self.object_points[cube_id] = np.float32((0, 0, 0)).reshape(3, 1)
        env.objects.append(cube_id)
        return cube_id

    def determine_task_stage(self, colormap=None, heightmap=None,
                             object_mask=None, visible_beads=None):
        """Get the task stage in a consistent manner among different policies.

        When training an oracle policy, we can determine the training stage,
        which is critical because of this task's particular quirks in
        requiring different action parameters (particularly height of the
        pull) for each stage. One option is to use this method to determine
        the hard-coded task stage for each task. This does depend on the
        learned policy inferring when to switch among task stages?

        For policies, I use simpler methods in their classes. They don't call this.

        Returns: False if the task is almost certainly going to fail.
        """
        if self.task_stage == 2 and (len(self.items_in_bag_IDs) == len(self.item_IDs)):
            #print('now on task stage 2 --> 3')
            self.task_stage = 3
            return (True, None)
        elif self.task_stage == 3:
            return (True, None)

        # Hand-tuned, seems reasonable to use.
        BUF = 0.025

        # Check object_mask for all IDs that correspond to the cable ring.
        cable_IDs = np.array(self.cable_bead_IDs)
        bead_mask = np.isin(object_mask, test_elements=cable_IDs)

        # Threshold image to get 0s and 255s (255s=bead pixels) and find its contours.
        bead_mask = np.uint8(bead_mask * 255)
        contours, _ = cv2.findContours(bead_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If only a few beads are visible (or no contours detected) exit early.
        frac_visible = len(visible_beads) / len(self.cable_bead_IDs)
        if len(contours) == 0 or frac_visible <= BEAD_THRESH:
            return (False, None)

        # Combine contours via concatenation (shape=(N,1,2)) and get the convex hull.
        allc = np.concatenate([c for c in contours])
        contours_list = [allc]
        hull_list = [cv2.convexHull(c) for c in contours_list]

        # Make an RGB image, then draw the filled-in area of all items in `hull_list`.
        hull = np.zeros((bead_mask.shape[0], bead_mask.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(hull, hull_list, -1, (255,255,255), thickness=-1)
        hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)

        # Following task.random_pose, use object_size to find placing points. Assumes
        # object sizes are same. We add a buffer since convex hulls inflate area.
        object_size = self.item_sizes[0]
        object_size = (object_size[0] + BUF, object_size[1] + BUF, object_size[2])
        max_size = np.sqrt(object_size[0]**2 + object_size[1]**2)
        erode_size = int(np.round(max_size / self.pixel_size))

        # Use cv2.erode to find place pixels on the hull, converted to grayscale.
        place_pixels = np.uint8(hull == 255)
        kernel = np.ones((erode_size, erode_size), np.uint8)
        place_pixels_eroded = cv2.erode(place_pixels, kernel)

        # On stage 1, if there exists any possible placing point, go to stage 2.
        assert self.task_stage == 1
        if np.sum(place_pixels_eroded) > 0:
            self.task_stage = 2
            #print('now on task stage 1 --> 2')
        return (True, place_pixels_eroded)


class BagItemsHard(BagEnv):
    """The harder version of BagItemsEasy, where we randomize the items."""

    def __init__(self):
        super().__init__()
        self.max_steps = 9+1
        self.metric = 'bag-items'
        self._name = 'bag-items-hard'
        self._targets_visible = False
        self._debug = False

        # Can make this smaller compared to bag-items-alone.
        self.num_force_iters = 8

        # Extra items, in addition to the bag.
        self._nb_items = 2
        self._max_total_dims = 0.070  # reduced from 0.08 for training

        # Env reference so we can call Task.get_object_masks(env)
        self.env = None

        # Exactly the same as BagItemsEasy.
        self.primitive_params = {
            1: {'speed': 0.004,
                'delta_z': -0.001,
                'prepick_z': 0.10,
                'postpick_z': 0.05,
                'preplace_z': 0.05,
                'pause_place': 0.25,
            },
            2: {'speed': 0.010,
                'delta_z': -0.0005, # same as cloth-cover
                'prepick_z': 0.10,  # hopefully makes it faster
                'postpick_z': 0.30,
                'preplace_z': 0.30,
                'pause_place': 0.0,
            },
            3: {'speed': 0.002,  # Will this slow bag movement?
                'delta_z': -0.001,
                'prepick_z': 0.08,  # hopefully makes it faster
                'postpick_z': 0.40,
                'preplace_z': 0.40,
                'pause_place': 0.50,  # used to be 2, maybe make smaller?
            },
        }
        self.task_stage = 1

    def reset(self, env):
        self.total_rewards = 0
        self.object_points = {}
        self.t = 0
        self.task_stage = 1
        self.cable_bead_IDs = []
        self.def_IDs = []
        self.env = env
        self.exit_gracefully = False

        # New stuff versus bag-alone-open, to better track stats.
        self.item_IDs = []
        self.items_in_bag_IDs = []
        self.item_sizes = []

        # Add square target zone and determine its corners.
        self.add_zone(env)
        self.get_target_zone_corners()

        # Pose of the bag, sample mid-air to let it drop naturally.
        bpos, _ = self.random_pose(env, self._bag_size)
        self.base_pos = [bpos[0], bpos[1], self._drop_height]
        self.base_orn = self._sample_bag_orientation()

        # Add the bag, load info about top ring, and make a cable.
        self.bag_id = self.add_bag(env, self.base_pos, self.base_orn)
        self.understand_bag_top_ring(env, self.base_pos)
        self.add_cable_ring(env)

        # Add randomly-shaped boxes.
        for _ in range(self._nb_items):
            box_id, box_size = self.add_random_box(env, self._max_total_dims)
            self.item_IDs.append(box_id)
            self.item_sizes.append(box_size)

        # Env must begin before we can apply forces.
        if self._debug:
            self.debug()
        env.start()

        # Add a small force to perturb the bag.
        self._apply_small_force(num_iters=self.num_force_iters)
        time.sleep(self._settle_secs)

        # Check that all added blocks are visible.
        colormap, heightmap, object_mask = self.get_object_masks(env)
        for ID in self.item_IDs:
            if ID not in object_mask:
                print(f'Warning, ID={ID} not in object_mask during reset(). Exit!')
                self.exit_gracefully = True
                self.save_images(colormap, heightmap, object_mask)
        env.pause()

    def add_random_box(self, env, max_total_dims):
        """Generate randomly shaped box, from aligning env.

        Make rand_x and rand_y add up to the max_total. Also, the aligning
        env uses a box with mass 0.1, but we can make ours lighter. But, will
        it cause the block to bounce too much when inserted in the bag?

        Also returning the object size, so we can use it for later.

        max_total_dims: Use to control how long we can make boxes. I would
            keep this value at a level making these boxes comparable to the
            cubes, if not smaller, used in bag-items-easy.
        """
        min_val = 0.015
        assert min_val*2 <= max_total_dims, min_val
        rand_x = np.random.uniform(min_val, max_total_dims - min_val)
        rand_y = max_total_dims - rand_x
        box_size = (rand_x, rand_y, 0.03)

        box_pose = self.random_pose(env, box_size)
        box_template = 'assets/box/box-template.urdf'
        box_urdf = self.fill_template(box_template, {'DIM': box_size})
        box_id = env.add_object(box_urdf, box_pose)
        os.remove(box_urdf)
        self.color_random_brown(box_id)
        self.object_points[box_id] = np.float32((0, 0, 0)).reshape(3, 1)
        self._IDs[box_id] = 'random_box'
        return (box_id, box_size)

    def determine_task_stage(self, colormap=None, heightmap=None,
                             object_mask=None, visible_beads=None):
        """Get the task stage in a consistent manner among different policies.

        When training an oracle policy, we can determine the training stage,
        which is critical because of this task's particular quirks in
        requiring different action parameters (particularly height of the
        pull) for each stage. One option is to use this method to determine
        the hard-coded task stage for each task. This does depend on the
        learned policy inferring when to switch among task stages?

        For policies, I use simpler methods in their classes. They don't call this.

        Returns: False if the task is almost certainly going to fail.
        """
        if self.task_stage == 2 and (len(self.items_in_bag_IDs) == len(self.item_IDs)):
            #print('now on task stage 2 --> 3')
            self.task_stage = 3
            return (True, None)
        elif self.task_stage == 3:
            return (True, None)

        # Hand-tuned, if too small the agent won't open the bag enough ...
        BUF = 0.025

        # But we can decrease it if we're on task stage 2 and have to put in more items.
        if self.task_stage == 2:
            BUF = 0.015

        # Check object_mask for all IDs that correspond to the cable ring.
        cable_IDs = np.array(self.cable_bead_IDs)
        bead_mask = np.isin(object_mask, test_elements=cable_IDs)

        # Threshold image to get 0s and 255s (255s=bead pixels) and find its contours.
        bead_mask = np.uint8(bead_mask * 255)
        contours, _ = cv2.findContours(bead_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If only a few beads are visible (or no contours detected) exit early.
        frac_visible = len(visible_beads) / len(self.cable_bead_IDs)
        if len(contours) == 0 or frac_visible <= BEAD_THRESH:
            return (False, None)

        # Combine contours via concatenation (shape=(N,1,2)) and get the convex hull.
        allc = np.concatenate([c for c in contours])
        contours_list = [allc]
        hull_list = [cv2.convexHull(c) for c in contours_list]

        # Make an RGB image, then draw the filled-in area of all items in `hull_list`.
        hull = np.zeros((bead_mask.shape[0], bead_mask.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(hull, hull_list, -1, (255,255,255), thickness=-1)
        hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)

        # Following task.random_pose, use object_size to find placing points. Assumes
        # object sizes are same. We add a buffer since convex hulls inflate area.
        # TODO(daniel) this is where we should really be using rotation.
        object_size = self.item_sizes[0]
        object_size = (object_size[0] + BUF, object_size[1] + BUF, object_size[2])
        max_size = np.sqrt(object_size[0]**2 + object_size[1]**2)
        erode_size = int(np.round(max_size / self.pixel_size))

        if self.task_stage == 2 and len(self.items_in_bag_IDs) > 0:
            # For the hard bag-items version, get array of 0s = items in bag (hence,
            # invalid placing points) and 1s = all other points (could be valid).
            pixels_bag_items = np.ones((hull.shape[0], hull.shape[1]), dtype=np.uint8)
            for item_id in self.items_in_bag_IDs:
                item_pix = np.uint8(item_id == object_mask)
                pixels_bag_items = pixels_bag_items & item_pix  # Logical AND
            pixels_no_bag_items = np.uint8(1 - pixels_bag_items)
        else:
            # Make it all 1s so it's safe to apply logical AND with hull pixels.
            pixels_no_bag_items = np.ones((hull.shape[0], hull.shape[1]), dtype=np.uint8)

        # Combine the hull and pixel conditions.
        place_pixels_hull = np.uint8(hull == 255)
        place_pixels = place_pixels_hull & pixels_no_bag_items

        # Use cv2.erode to find valid place pixels.
        kernel = np.ones((erode_size, erode_size), np.uint8)
        place_pixels_eroded = cv2.erode(place_pixels, kernel)

        # If we're in task stage 2 and there's nothing, let's revert back to original.
        if self.task_stage == 2 and np.sum(place_pixels_eroded) == 0:
            place_pixels_eroded = cv2.erode(place_pixels_hull, kernel)

        # Keep this debugging code to make it easier to inspect.
        if self._debug:
            heightmap = heightmap / np.max(heightmap) * 255
            place_rgb = cv2.cvtColor(hull.copy(), cv2.COLOR_GRAY2BGR)
            place_rgb[place_pixels_eroded > 0] = 127  # gray
            print(f'max_size: {max_size:0.3f}, erode_size: {erode_size}')
            print(f'number of pixels for placing: {np.sum(place_pixels)}')
            print(f'number of pixels for placing (after eroding): {np.sum(place_pixels_eroded)}')
            nb = len([x for x in os.listdir('tmp/') if 'color' in x and '.png' in x])
            cv2.imwrite(f'tmp/img_{nb}_colormap.png', cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR).astype(np.uint8))
            cv2.imwrite(f'tmp/img_{nb}_heightmap.png', heightmap.astype(np.uint8))
            cv2.imwrite(f'tmp/img_{nb}_bead_mask.png', bead_mask)
            cv2.imwrite(f'tmp/img_{nb}_place_rgb.png', cv2.cvtColor(place_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'tmp/img_{nb}_place_pixels_eroded.png', np.uint8(place_pixels_eroded * 255))
            if self.task_stage == 2 and len(self.items_in_bag_IDs) > 0:
                pixels_no_bag_items *= 255
                cv2.imwrite(f'tmp/img_{nb}_pixels_no_bag_items.png', np.uint8(pixels_no_bag_items))

        # If on stage 1, and there exists any possible placing point, go to stage 2.
        if self.task_stage == 1:
            if np.sum(place_pixels_eroded) > 0:
                self.task_stage = 2
                #print('now on task stage 1 --> 2')
        return (True, place_pixels_eroded)


class BagColorGoal(BagEnv):
    """A goal-based version of bags which doesn't require goal-conditioned PICKING."""

    def __init__(self):
        super().__init__()
        self.max_steps = 8+1
        self.metric = 'bag-color-goal'
        self._name = 'bag-color-goal'
        self._targets_visible = False
        self._debug = False

        # Can make this smaller compared to bag-items-alone.
        self.num_force_iters = 8

        # Size of the items we should be sampling.
        self.max_total_dims = 0.100

        # Env reference so we can call Task.get_object_masks(env)
        self.env = None

        # New stuff, have colors and need to randomly sample one later.
        self.colors_to_sample = ['purple', 'blue', 'yellow', 'red', 'green', 'cyan']
        self.num_bags = 2

        # Exactly the same as BagItemsEasy. However we don't need the
        # bag pulling upwards portion (thankfully).
        self.primitive_params = {
            1: {'speed': 0.004,
                'delta_z': -0.001,
                'prepick_z': 0.10,
                'postpick_z': 0.05,
                'preplace_z': 0.05,
                'pause_place': 0.25,
            },
            2: {'speed': 0.010,
                'delta_z': -0.0005, # same as cloth-cover
                'prepick_z': 0.10,  # hopefully makes it faster
                'postpick_z': 0.30,
                'preplace_z': 0.30,
                'pause_place': 0.0,
            },
        }
        self.task_stage = 1

        # Ah for this we actually should override the bag scales.
        self.beads_per_bag = 32
        self._bag_scale = 0.10
        self._bag_size = (1. * self._bag_scale, 1. * self._bag_scale, 0.01)
        self._mass = 1.0
        self._scale = 0.25
        self._collisionMargin = 0.003

    def reset(self, env, last_info=None):
        """How to use last_info, actually? That's an interesting Q.

        Maybe it's not needed in this case since we don't have a pose-based metric ...?
        However, then we don't really need a goal-based image. Oh, this is a different
        flavor. The goal comes from the target image. Unlike in 'sorting', the color of
        the item that has to go in the zone will be different.

        However, we have to use last_info to define what the success criteria should be,
        right? In particular, we need to save the actual target color, right? Yes, for
        that please use `last_info['target_block_color']`.

        NO TARGET ZONE! However, assume the demonstrator and any reward function (for
        any policy) can access: `self.target_block_color` and `self.target_block_ID`.
        """
        self.total_rewards = 0
        self.object_points = {}
        self.t = 0
        self.task_stage = 1
        self.env = env
        self.exit_gracefully = False
        self._IDs = {}

        # Unlike other tasks, these now have to contain info about multiple bags.
        self.cable_bead_IDs = []        # first 32 for first bag, next 32 for second bag, etc.
        self.def_IDs = []               # exactly the bag IDs.
        self.bag_colors = []            # store so we can load properly.
        self.bag_base_pos = []          # store so we can load properly.
        self.bag_base_orn = []          # store so we can load properly.
        self.block_color = None         # we'll store this, but it's not critical.
        self.single_block_ID = None     # actually this is important -- we can detect if we're on task stage 2 with this.
        self.single_block_size = None   # used for erosion for demonstrator
        self.cable_bead_target_bag_IDs = None

        # Sample bag colors, make them unique. Store in bag_colors for later.
        color_indices = np.random.choice(len(self.colors_to_sample), size=self.num_bags, replace=False)

        # -------------------------------------------------------------------------------------------------- #
        # TARGET BAG! Must handle the saving / loading if needed.
        # NOTE! we do need the target bag to be in the same spatial position as it was during training.
        # Unfortunately it is restrictive, but hopefully we can figure this out in follow-up work.
        # -------------------------------------------------------------------------------------------------- #
        if last_info is not None:
            base_pos = last_info['bag_base_pos']               # see note above
            base_orn = last_info['bag_base_orn']               # see note above
            #bpos, _ = self.random_pose(env, self._bag_size)   # can comment out and replace with last_info?
            #base_pos = [bpos[0], bpos[1], self._drop_height]  # can comment out and replace with last_info?
            #base_orn = self._sample_bag_orientation()         # can comment out and replace with last_info?
            bag_color = last_info['bag_target_color']       # must make sure distractor bag isn't bag color ...
        else:
            bpos, _ = self.random_pose(env, self._bag_size)
            base_pos = [bpos[0], bpos[1], self._drop_height]
            base_orn = self._sample_bag_orientation()
            bag_color = self.colors_to_sample[ color_indices[0] ]
        self.bag_base_pos.append(base_pos)
        self.bag_base_orn.append(base_orn)

        # Actually add the bag, followed by its ring. Note: def_IDs is added in `add_bag`.
        bag_id = self.add_bag(env, base_pos, base_orn, bag_color=bag_color)
        self.understand_bag_top_ring(env, base_pos)
        self.add_cable_ring(env, bag_id=bag_id)
        self.bag_colors.append(bag_color)

        # ------------------------------------------------------------------------------------------------ #
        # Create ONE block. Hopefully policy (a) opens the RIGHT bag, (b) inserts this thing in it!
        # Putting this first to hopefully avoid bags on top of blocks ... maybe ?
        # ------------------------------------------------------------------------------------------------ #
        b_int = int(np.random.randint(len(self.colors_to_sample)))
        self.block_color = self.colors_to_sample[b_int]
        bname = f'block_{self.block_color}'
        self.single_block_ID, self.single_block_size = self.add_block(env, bname, self.max_total_dims)
        p.changeVisualShape(self.single_block_ID, -1, rgbaColor=U.COLORS[self.block_color]+[1])

        # ------------------------------------------------------------------------------------------------ #
        # DISTRACTOR BAG. MUST USE A DIFFERENT COLOR compared to the target bag. Ahhh ... we have to make
        # sure we don't overwrite anything that might be used for the demonstrator code, which uses stuff
        # from `understand_bag_top_ring`. Ah, easiest way seems to be just to not call that method :D
        # ------------------------------------------------------------------------------------------------ #
        bpos, _ = self.random_pose(env, self._bag_size)
        base_pos = [bpos[0], bpos[1], self._drop_height]
        base_orn = self._sample_bag_orientation()
        self.bag_base_pos.append(base_pos)
        self.bag_base_orn.append(base_orn)

        # Fetch and store the bag color, redrawing as needed (since if loading we ignore the b=0 color...).
        bag_color = self.colors_to_sample[ color_indices[1] ]
        while bag_color == self.bag_colors[0]:
            bag_color = self.colors_to_sample[ int(np.random.randint(len(self.colors_to_sample))) ]
        self.bag_colors.append(bag_color)

        # Actually add the bag, followed by its ring. Note: def_IDs is added in `add_bag`.
        bag_id = self.add_bag(env, base_pos, base_orn, bag_color=bag_color)
        #self.understand_bag_top_ring(env, base_pos)  # see note above
        self.add_cable_ring(env, bag_id=bag_id)
        assert bag_id in self.def_IDs, self.def_IDs

        # Bells and whistles. Let's make a cable bead only for targets.
        assert len(self.cable_bead_IDs) == 2 * self.beads_per_bag, len(self.cable_bead_IDs)
        self.cable_bead_target_bag_IDs = self.cable_bead_IDs[:self.beads_per_bag]

        # Env must begin before we can apply forces.
        if self._debug:
            self.debug()
        env.start()

        # Add a small force to perturb the bag.
        self._apply_small_force(num_iters=self.num_force_iters)
        time.sleep(self._settle_secs)

        # Check that all added blocks are visible.
        colormap, heightmap, object_mask = self.get_object_masks(env)
        if self.single_block_ID not in object_mask:
            print(f'Warning, ID={self.single_block_ID} not in object_mask during reset(). Exit!')
            self.exit_gracefully = True
            self.save_images(colormap, heightmap, object_mask)
        else:
            #print(f'Note, ID={self.single_block_ID} in object_mask during reset()!')
            #self.save_images(colormap, heightmap, object_mask)
            pass
        env.pause()

    def add_block(self, env, block_name, max_total_dims):
        """Generate randomly shaped block for the goal based task.

        Similar to the random block method used for bag-items-hard.
        env uses a box with mass 0.1, but we can make ours lighter. But, will
        it cause the block to bounce too much when inserted in the bag?

        Also returning the object size, so we can use it for later.
        """
        #min_val = 0.025
        #assert min_val*2 <= max_total_dims, min_val
        #rand_x = np.random.uniform(min_val, max_total_dims - min_val)
        #rand_y = max_total_dims - rand_x
        #box_size = (rand_x, rand_y, 0.03)
        box_size = (0.045, 0.045, 0.030)  # may make it less susceptible to falling out

        box_pose = self.random_pose(env, box_size)
        box_template = 'assets/box/box-template.urdf'
        box_urdf = self.fill_template(box_template, {'DIM': box_size})
        box_id = env.add_object(box_urdf, box_pose)
        os.remove(box_urdf)
        self.color_random_brown(box_id)
        self.object_points[box_id] = np.float32((0, 0, 0)).reshape(3, 1)
        self._IDs[box_id] = block_name
        return (box_id, box_size)

    def is_item_in_bag(self):
        """Used to determine reward. We want item to be inside the bag hull.

        Actually, first detect bead visibility. If we can't see any then we should
        just return 0 because that means we've messed up somewhere.

        Returns an info dict, which has 'reward' keyword.
        """
        result = {'exit_early': False,
                  'frac_in_target_bag': 0.0,
                  'frac_in_distract_bag': 0.0}
        colormap, heightmap, object_mask = self.get_object_masks(self.env)

        # Detect visible beads for the FIRST bag! The second bag is irrelevant.
        visible_beads = []
        for bead in self.cable_bead_target_bag_IDs:
            if bead in object_mask:
                visible_beads.append(bead)

        # Check object_mask for all IDs that correspond to the cable ring.
        cable_IDs = np.array(self.cable_bead_target_bag_IDs)
        bead_mask = np.isin(object_mask, test_elements=cable_IDs)

        # Threshold image to get 0s and 255s (255s=bead pixels) and find its contours.
        bead_mask = np.uint8(bead_mask * 255)
        contours, _ = cv2.findContours(bead_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If only a few beads are visible (or no contours detected) exit early.
        # NOTE: KEEP THE KEYS CONSISTENT WITH THIS AND THE RESULT!
        frac_visible = len(visible_beads) / len(self.cable_bead_target_bag_IDs)
        result['beads_visible'] = len(visible_beads)
        if len(contours) == 0 or frac_visible <= BEAD_THRESH:
            result['exit_early'] = True
            return result

        # Combine contours via concatenation (shape=(N,1,2)) and get the convex hull.
        allc = np.concatenate([c for c in contours])
        contours_list = [allc]
        hull_list = [cv2.convexHull(c) for c in contours_list]

        # Make an RGB image, then draw the filled-in area of all items in `hull_list`.
        hull = np.zeros((bead_mask.shape[0], bead_mask.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(hull, hull_list, -1, (255,255,255), thickness=-1)
        hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)

        # (now comes the new stuff for the reward vs the task determination stage)
        # next, find the pixels of the target item, and determine fraction that coincide w/the target hull.
        pixels_target_item    = np.uint8(object_mask == self.single_block_ID)
        target_and_hull       = pixels_target_item & hull  # intersection of target item's and hull's pixels
        target_and_hull_count = np.count_nonzero(target_and_hull)  # number of pixels for this intersection
        target_pixel_count    = np.count_nonzero(pixels_target_item)  # number of pixels for target item
        frac_in_target_bag    = target_and_hull_count / (float(target_pixel_count) + 0.01)

        # ----------------------------------------------------------------------------- #
        # DISTRACTORS: now check hull of the OTHER bag ... ONLY for debugging purposes. #
        # ----------------------------------------------------------------------------- #
        visible_beads = []
        for bead in self.cable_bead_IDs[self.beads_per_bag:]:
            if bead in object_mask:
                visible_beads.append(bead)

        # Check object_mask for all IDs that correspond to the cable ring.
        cable_IDs = np.array( self.cable_bead_IDs[self.beads_per_bag:] )
        bead_mask = np.isin(object_mask, test_elements=cable_IDs)

        # Threshold image to get 0s and 255s (255s=bead pixels) and find its contours.
        bead_mask = np.uint8(bead_mask * 255)
        contours, _ = cv2.findContours(bead_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If only a few beads are visible (or no contours detected) then this fraction is 0.
        frac_visible = len(visible_beads) / self.beads_per_bag
        if len(contours) == 0 or frac_visible <= BEAD_THRESH:
            frac_in_distract_bag = 0.0
        else:
            # Combine contours via concatenation (shape=(N,1,2)) and get the convex hull.
            allc = np.concatenate([c for c in contours])
            contours_list = [allc]
            hull_list = [cv2.convexHull(c) for c in contours_list]

            # Make an RGB image, then draw the filled-in area of all items in `hull_list`.
            hull = np.zeros((bead_mask.shape[0], bead_mask.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(hull, hull_list, -1, (255,255,255), thickness=-1)
            hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)

            # We use pixels_target_item from earlier, just a different hull now.
            target_and_hull       = pixels_target_item & hull  # intersection of target item's and hull's pixels
            target_and_hull_count = np.count_nonzero(target_and_hull)  # number of pixels for this intersection
            target_pixel_count    = np.count_nonzero(pixels_target_item)  # number of pixels for target item
            frac_in_distract_bag  = target_and_hull_count / (float(target_pixel_count) + 0.01)

        # TODO(daniel) ensure that this matches the keys from exiting early!
        #result['frac_distract_in_hull'] = frac_distract_in_hull
        result['frac_in_target_bag']   = frac_in_target_bag
        result['frac_in_distract_bag'] = frac_in_distract_bag
        return result

        ##place_rgb = cv2.cvtColor(hull.copy(), cv2.COLOR_GRAY2BGR)
        ##place_rgb[pixels_target_item > 0] = 127  # gray for pixel targets
        ##nb = len([x for x in os.listdir('tmp/') if 'color' in x and '.png' in x])
        ##cv2.imwrite(f'tmp/img_{nb}_colormap.png', cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR).astype(np.uint8))
        ##cv2.imwrite(f'tmp/img_{nb}_bead_mask.png', bead_mask)
        ##cv2.imwrite(f'tmp/img_{nb}_place_rgb.png', cv2.cvtColor(place_rgb, cv2.COLOR_RGB2BGR))

    def determine_task_stage(self, colormap=None, heightmap=None,
                             object_mask=None, visible_beads=None):
        """Get the task stage in a consistent manner among different policies.

        This should be easier as compared to the bag-items since we only handle
        two task stages (and we don't have drastically different action params).
        """
        if self.task_stage == 2:
            print('ON TASK STAGE 2, still here, should not normally happen...')
            return (True, None)

        # Hand-tuned, if too small the agent won't open the bag enough ...
        BUF = 0.025

        # Check object_mask for all IDs that correspond to the cable ring OF THE TARGET!
        cable_IDs = np.array(self.cable_bead_target_bag_IDs)
        bead_mask = np.isin(object_mask, test_elements=cable_IDs)

        # Threshold image to get 0s and 255s (255s=bead pixels) and find its contours.
        bead_mask = np.uint8(bead_mask * 255)
        contours, _ = cv2.findContours(bead_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If only a few beads are visible (or no contours detected) exit early.
        frac_visible = len(visible_beads) / len(self.cable_bead_target_bag_IDs)
        if len(contours) == 0 or frac_visible <= BEAD_THRESH:
            return (False, None)

        # Combine contours via concatenation (shape=(N,1,2)) and get the convex hull.
        allc = np.concatenate([c for c in contours])
        contours_list = [allc]
        hull_list = [cv2.convexHull(c) for c in contours_list]

        # Make an RGB image, then draw the filled-in area of all items in `hull_list`.
        hull = np.zeros((bead_mask.shape[0], bead_mask.shape[1], 3), dtype=np.uint8)
        cv2.drawContours(hull, hull_list, -1, (255,255,255), thickness=-1)
        hull = cv2.cvtColor(hull, cv2.COLOR_BGR2GRAY)

        # Following task.random_pose, use object_size to find placing points. Assumes
        # object sizes are same. We add a buffer since convex hulls inflate area.
        # TODO(daniel) this is where we should really be using rotation.
        object_size = self.single_block_size
        object_size = (object_size[0] + BUF, object_size[1] + BUF, object_size[2])
        max_size = np.sqrt(object_size[0]**2 + object_size[1]**2)
        erode_size = int(np.round(max_size / self.pixel_size))

        # Use cv2.erode to find place pixels on the hull, converted to grayscale.
        place_pixels = np.uint8(hull == 255)
        kernel = np.ones((erode_size, erode_size), np.uint8)
        place_pixels_eroded = cv2.erode(place_pixels, kernel)

        # On stage 1, if there exists any possible placing point, go to stage 2.
        assert self.task_stage == 1
        if np.sum(place_pixels_eroded) > 0:
            self.task_stage = 2
            #print('now on task stage 1 --> 2')
        return (True, place_pixels_eroded)
