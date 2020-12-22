#!/usr/bin/env python
"""
Daniel's set of environments related to cables. Remember:

Interpreting self.zone_pose[0] = (x,y,z) where x and y are the vertical and
horizontal (resp.) ranges in the diagram:

  -0.5 .... +0.5
  ------------  0.3
  |          |   :
  |          |   :
  |          |   :
  ------------  0.7

The self.zone_pose corresponds to the colored yellow lines in the GUI. The
(0,0,0) location corresponds to the base of the robot. Unlike cloth and bags,
we don't need the soft body here, so for the most part this is using code
directly from Andy.

Normally cable uses at most 20 actions but hopefully we can reduce that.
"""
import os
import time
import numpy as np
import pybullet as p
from ravens.tasks import Task
from ravens import utils as U


class CableEnv(Task):
    """Superclass for cables / rope / Deformable Linear Objects envs."""

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.primitive = 'pick_place'
        self.max_steps = 11
        self._IDs = {}
        self._debug = False
        self._settle_secs = 2

        # The usual zone stuff (see dan_cloth and dan_bags)
        self._zone_scale = 0.01
        self._zone_length = (20.0 * self._zone_scale)
        self.zone_size = (20.0 * self._zone_scale, 20.0 * self._zone_scale, 0)

        # Cable-related parameters, can override in subclass.
        self.num_parts = 20
        self.radius = 0.005
        self.length = 2 * self.radius * self.num_parts * np.sqrt(2)
        self.colors = [U.COLORS['blue']]

        # Put cable bead IDs here, so we don't count non cable IDs for targets.
        self.cable_bead_IDs = []

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
        return zone_id

    def add_cable(self, env, size_range, info={}, cable_idx=0,
            direction='z', max_force=100):
        """Add a cable like Andy does it in his cable environment.

        Add each `part_id` to (a) env.objects, (b) object_points, (c) _IDs,
        and (d) cable_bead_IDs. For (b) it is because, like the sweeping env,
        the demonstrator detects the one farthest from the goal to use as the
        pick, and `object_points` is also used to tally up items that are
        within the zone, to compute the reward, or the net % improvement, so
        if we add more cables, all beads in them need to be added to this
        dict. But I'm adding (d) so that we can distinguish between bead vs
        non-bead objects --- just in case!

        When iterating through the number of parts, ensure that the given
        cable is _separate_ from prior cables, in case there are more than
        one. ALL beads are put in the `env.objects` list.

        The zone_range is used because we need the cables to start outside of
        the zone. However, we should check if sampling multiple cables will
        work; there might not be space to sample a zone and two cables.

        Parameters
        ----------
        :size_range: Used to indicate the area of the target, so the beads
            avoid spawning there.
        :info: Stores relevant stuff, such as for ground-truth targets.
        :cable_idx: Only useful if we spawn multiple cables.
        :direction: Usually we want z, for verticality.

        Returns nothing, but could later return the bead IDs if needed.
        """
        num_parts = self.num_parts
        radius = self.radius
        length = self.length
        color = self.colors[cable_idx] + [1]
        color_end = U.COLORS['yellow'] + [1]

        # Add beaded cable.
        distance = length / num_parts
        position, _ = self.random_pose(env, size_range)
        position = np.float32(position)
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius]*3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius*1.5)

        # Iterate through parts and create constraints as needed.
        for i in range(num_parts):
            if direction == 'x':
                position[0] += distance
                parent_frame = (distance, 0, 0)
            elif direction == 'y':
                position[1] += distance
                parent_frame = (0, distance, 0)
            else:
                position[2] += distance
                parent_frame = (0, 0, distance)

            part_id = p.createMultiBody(0.1, part_shape, part_visual,
                    basePosition=position)
            if i > 0:
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=env.objects[-1],
                        parentLinkIndex=-1,
                        childBodyUniqueId=part_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=max_force)

            if (i > 0) and (i < num_parts - 1):
                p.changeVisualShape(part_id, -1, rgbaColor=color)
            elif i == num_parts - 1:
                p.changeVisualShape(part_id, -1, rgbaColor=color_end)

            # The usual for tracking IDs. Four things to add.
            self.cable_bead_IDs.append(part_id)
            self._IDs[part_id] = f'cable_part_{str(part_id).zfill(2)}'
            env.objects.append(part_id)
            self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

            # Get target placing positions for each cable bead, if applicable.
            if self._name == 'cable-shape' or self._name == 'cable-shape-notarget' or \
                    self._name == 'cable-line-notarget':
                # ----------------------------------------------------------- #
                # Here, zone_pose = square_pose, unlike Ravens cable, where the
                # zone_pose is shifted so that its center matches the straight
                # line segment center. For `true_position`, we use `zone_pose`
                # but apply the correct offset to deal with the sides. Note
                # that `length` is the size of a fully smoothed cable, BUT we
                # made a rectangle with each side <= length.
                # ----------------------------------------------------------- #
                lx = info['lengthx']
                ly = info['lengthy']
                r = radius

                if info['nb_sides'] == 1:
                    # Here it's just a straight line on the 'lx' side.
                    x_coord = lx / 2 - (distance * i)
                    y_coord = 0
                    true_position = (x_coord - r, y_coord, 0)

                elif info['nb_sides'] == 2:
                    # Start from lx side, go 'left' to the pivot point, then on
                    # the ly side, go 'upwards' but offset by `i`. For radius
                    # offset, I just got this by tuning. XD
                    if i < info['cutoff']:
                        x_coord = lx / 2 - (distance * i)
                        y_coord = -ly / 2
                        true_position = (x_coord - r, y_coord, 0)
                    else:
                        x_coord = -lx / 2
                        y_coord = -ly / 2 + (distance * (i - info['cutoff']))
                        true_position = (x_coord, y_coord + r, 0)

                elif info['nb_sides'] == 3:
                    # Start from positive lx, positive ly, go down to first
                    # pivot. Then go left to the second pivot, then up again.
                    # For v1, division by two is because we assume BOTH of the
                    # 'ly edges' were divided by two.
                    v1 = (self.num_parts - info['cutoff']) / 2
                    v2 = self.num_parts - v1
                    if i < v1:
                        x_coord = lx / 2
                        y_coord = ly / 2 - (distance * i)
                        true_position = (x_coord, y_coord - r, 0)
                    elif i < v2:
                        x_coord = lx / 2 - (distance * (i - v1))
                        y_coord = -ly / 2
                        true_position = (x_coord - r, y_coord, 0)
                    else:
                        x_coord = -lx / 2
                        y_coord = -ly / 2 + (distance * (i - v2))
                        true_position = (x_coord, y_coord + r, 0)

                elif info['nb_sides'] == 4:
                    # I think this is similar to the 2-side case: we start in
                    # the same direction and go counter-clockwise.
                    v1 = info['cutoff'] / 2
                    v2 = num_parts / 2
                    v3 = (num_parts + info['cutoff']) / 2
                    if i < v1:
                        x_coord = lx / 2 - (distance * i)
                        y_coord = -ly / 2
                        true_position = (x_coord, y_coord, 0)
                    elif i < v2:
                        x_coord = -lx / 2
                        y_coord = -ly / 2 + (distance * (i - v1))
                        true_position = (x_coord, y_coord, 0)
                    elif i < v3:
                        x_coord = -lx / 2 + (distance * (i - v2))
                        y_coord = ly / 2
                        true_position = (x_coord, y_coord, 0)
                    else:
                        x_coord = lx / 2
                        y_coord = ly / 2 - (distance * (i - v3))
                        true_position = (x_coord, y_coord, 0)

                # Map true_position onto the workspace from zone_pose.
                true_position = self.apply(self.zone_pose, true_position)

                # See `cable.py`: just get the places and steps set.
                self.goal['places'][part_id] = (true_position, (0, 0, 0, 1.))
                symmetry = 0
                self.goal['steps'][0][part_id] = (symmetry, [part_id])

                # Debugging target zones.
                if self.target_debug_markers:
                    sq_pose = ((true_position[0], true_position[1], 0.002), (0,0,0,1))
                    sq_template = 'assets/square/square-template-allsides-blue.urdf'
                    replace = {'DIM': (0.003,), 'HALF': (0.003 / 2,)}
                    urdf = self.fill_template(sq_template, replace)
                    env.add_object(urdf, sq_pose, fixed=True)
                    os.remove(urdf)
            else:
                print(f'Warning, env {self._name} will not have goals.')

    def add_cable_ring(self, env, info={}, cable_idx=0):
        """Add a cable, but make it connected at both ends to form a ring.

        For consistency, add each `part_id` to various information tracking
        lists and dictionaries (see `add_cable` documentation).

        :cable_idx: Used for environments with more than one cable.
        :info: Stores relevant stuff, such as for ground-truth targets.
        """
        def rad_to_deg(rad):
            return (rad * 180.0) / np.pi

        def get_discretized_rotations(num_rotations):
            # counter-clockwise
            theta = i * (2 * np.pi) / num_rotations
            return (theta, rad_to_deg(theta))

        # Bead properties.
        num_parts = self.num_parts
        radius = self.radius
        color = self.colors[cable_idx] + [1]

        # The `ring_radius` (not the bead radius!) has to be tuned somewhat.
        # Try to make sure the beads don't have notable gaps between them.
        ring_radius = info['ring_radius']
        beads = []
        bead_positions_l = []

        # Add beaded cable. Here, `position` is the circle center.
        position = np.float32(info['center_position'])
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius]*3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius*1.5)

        # Iterate through parts and create constraints as needed.
        for i in range(num_parts):
            angle_rad, _ = get_discretized_rotations(num_parts)
            px = ring_radius * np.cos(angle_rad)
            py = ring_radius * np.sin(angle_rad)
            #print(f'pos: {px:0.2f}, {py:0.2f}, angle: {angle_rad:0.2f}, {angle_deg:0.1f}')
            bead_position = np.float32([position[0] + px, position[1] + py, 0.01])
            part_id = p.createMultiBody(0.1, part_shape, part_visual,
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

            # Track beads.
            beads.append(part_id)
            bead_positions_l.append(bead_position)

            # The usual for tracking IDs. Four things to add.
            self.cable_bead_IDs.append(part_id)
            self._IDs[part_id] = f'cable_part_{str(part_id).zfill(2)}'
            env.objects.append(part_id)
            self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

            if 'cable-ring' in self._name:
                # For targets, we can assume the starting position gives us
                # that. Later, apply a random force on the ring to perturb it.
                true_position = (bead_position[0], bead_position[1], 0)
                self.goal['places'][part_id] = (true_position, (0, 0, 0, 1.))
                symmetry = 0
                self.goal['steps'][0][part_id] = (symmetry, [part_id])

                # Make the true positions visible if desired.
                if info['targets_visible']:
                    sq_pose = ((true_position[0], true_position[1], 0.002), (0,0,0,1))
                    sq_template = 'assets/square/square-template-allsides-green.urdf'
                    replace = {'DIM': (0.003,), 'HALF': (0.003 / 2,)}
                    urdf = self.fill_template(sq_template, replace)
                    env.add_object(urdf, sq_pose, fixed=True)
                    os.remove(urdf)
            else:
                print(f'Warning, env {self._name} will not have goals.')

    def add_block(self, env):
        """Add a block (really, a cube) from sorting env."""
        block_size = (0.04, 0.04, 0.04)
        block_pose = self.random_pose(env, block_size)
        block_urdf = 'assets/stacking/block.urdf'
        block_id = env.add_object(block_urdf, block_pose)
        self.object_points[block_id] = np.float32((0, 0, 0)).reshape(3, 1)
        self._IDs[block_id] = 'block'
        return block_id

    def add_random_box(self, env):
        """Generate randomly shaped box, from aligning env."""
        box_size = self.random_size(0.05, 0.15, 0.05, 0.15, 0.01, 0.06)
        box_pose = self.random_pose(env, box_size)
        box_template = 'assets/box/box-template.urdf'
        box_urdf = self.fill_template(box_template, {'DIM': box_size})
        box_id = env.add_object(box_urdf, box_pose)
        os.remove(box_urdf)
        self.color_random_brown(box_id)
        self.object_points[box_id] = np.float32((0, 0, 0)).reshape(3, 1)
        self._IDs[box_id] = 'random_box'
        return box_id

    def get_ID_tracker(self):
        return self._IDs

    def debug(self):
        np.set_printoptions(suppress=True, edgeitems=10, linewidth=200)
        print('\nInside {} reset()'.format(self._name))
        for ID in self.cable_bead_IDs:
            pose = U.round_pose(self.goal['places'][ID])
            print(f'Bead {str(ID).zfill(2)}, target: {pose}')
        print('Waiting {} secs for stuff to settle ...\n'.format(self._settle_secs))

    @property
    def circle_area(self):
        """Only if we are using the cable-ring environment..."""
        return np.pi * self.ring_radius ** 2

    @property
    def area_thresh(self):
        """Only if we are using the cable-ring environment...

        So far I think using 0.8 or higher might be too hard because moving
        the ring to the target can cause other areas to decrease.
        """
        return 0.75


class CableShape(CableEnv):
    """A single cable, and manipulating to a complex target.

    Application inspiration: moving a cable towards a target is commonly done
    in cases such as knot-tying and rearranging stuff on a surface, and more
    generally it's a common robotics benchmark.

    For now we are using targets based on line segments stacked with each
    other. This means we have to change the normal zone metric because it
    assumes a linear target, but shouldn't be too difficult. Also, because
    this involves just a simple cable, we are going to use the same
    pick_place demonstrator, so we need a `self.goal`.

    Remember that the UR5 is at zone (0,0,0) and the 'square' is like this:

      |     |   xxxx
      |  o  |   xxxx
      |     |   xxxx
      -------

    where `o` is the center of the robot, and `x`'s represent the workspace
    (horizontal axis is x). Then the 'line' has to fill in the top part of
    the square. Each edge has length `length` in code. We generalize this for
    targets of between 1 and 4 connected line segments. Use `length_x` and
    `length_y` to determine the lengths of the sides. With two sides, we get:

      |         xxxx
      |  o      xxxx  length_y
      |         xxxx
      -------
      length_x

    where `length_x + length_y = length`, or one of the original square
    sides. Thus, the square essentially defines where the target can be
    sampled. Also keep in mind that we actually use a rectangle, not a
    square; the square_pose is just used for a pose and sampling bounds.
    """

    def __init__(self):
        super().__init__()
        self.ee = 'suction'
        self.max_steps = 21
        self.metric = 'cable-target'
        self.primitive = 'pick_place'
        self._name = 'cable-shape'
        self._debug = False

        # Target zone and the debug marker visibility, usually (T, F).
        self.target_zone_visible = True
        self.target_debug_markers = False

        # Cable parameters. For this I think we want more than 20.
        self.num_parts = 24
        self.radius = 0.005
        self.length = 2 * self.radius * self.num_parts * np.sqrt(2)
        self.num_sides_low = 2
        self.num_sides_high = 4

        # Parameters for pick_place primitive.
        self.primitive_params = {
            1: {'speed': 0.001,
                'delta_z': -0.001,
                'postpick_z': 0.04,
                'preplace_z': 0.04,
                'pause_place': 0.0,
            },
        }
        self.task_stage = 1

        # To see if performance varies as a function of the number of sides.
        self.nb_sides = None

    def reset(self, env):
        self.total_rewards = 0
        self.object_points = {}
        self.t = 0
        self.task_stage = 1
        self.cable_bead_IDs = []

        # We need this to use the built-in pick_place demonstrator in `task.py`.
        self.goal = {'places': {}, 'steps': [{}]}

        # Sample the 'square pose' which is the center of a rectangle.
        square_size = (self.length, self.length, 0)
        square_pose = self.random_pose(env, square_size)
        #square_pose = (square_pose[0], (0,0,0,1)) # debugging only
        assert square_pose is not None, f'Cannot sample a pose.'

        # Be careful. We deduce ground-truth pose labels from zone_pose.
        self.zone_pose = square_pose
        zone_range = (self.length / 10, self.length / 10, 0)

        # Sample the number of sides to preserve from the rectangle.
        low, high = self.num_sides_low, self.num_sides_high
        self.nb_sides = nb_sides = np.random.randint(low=low, high=high+1) # note +1
        template = f'assets/rectangle/rectangle-template-sides-{nb_sides}.urdf'

        if nb_sides == 1:
            # One segment target: a straight cable should be of length `length`.
            lengthx = self.length
            lengthy = 0
            cutoff = 0
        elif nb_sides == 2:
            # Two segment target: length1 + length2 should equal a straight cable.
            cutoff = np.random.randint(0, self.num_parts + 1)
            alpha = cutoff / self.num_parts
            lengthx = self.length * alpha
            lengthy = self.length * (1 - alpha)
        elif nb_sides == 3:
            # Three segment target: remove length1, but need to remove a bit more.
            offset = 4  # avoid 'extremes'
            cutoff = np.random.randint(offset, self.num_parts + 1 - offset)
            alpha = cutoff / self.num_parts
            lengthx = self.length * alpha
            lengthy = (self.length * (1 - alpha)) / 2
        elif nb_sides == 4:
            # Four segment target, divide by two to make the cable 'fit'.
            offset = 4  # avoid 'extremes'
            cutoff = np.random.randint(offset, self.num_parts + 1 - offset)
            alpha = cutoff / self.num_parts
            lengthx = (self.length * alpha) / 2
            lengthy = (self.length * (1 - alpha)) / 2

        # I deduced DIM & HALF from rectangle template through trial & error.
        DIM = (lengthx, lengthy)
        HALF = (DIM[1] / 2, DIM[0] / 2)
        if self.target_zone_visible:
            replace = {'DIM': DIM, 'HALF': HALF}
            urdf = self.fill_template(template, replace)
            env.add_object(urdf, square_pose, fixed=True)
            os.remove(urdf)

        # Add cable.
        info = {'nb_sides': nb_sides, 'cutoff': cutoff, 'lengthx': lengthx,
                'lengthy': lengthy, 'DIM': DIM, 'HALF': HALF,}
        self.add_cable(env, size_range=zone_range, info=info)

        if self._debug:
            self.debug()
        env.start()
        time.sleep(self._settle_secs)
        env.pause()


class CableShapeNoTarget(CableShape):
    """CableShape, but without a target, so we need a goal image."""

    def __init__(self):
        super().__init__()
        self._name = 'cable-shape-notarget'
        self._debug = False

        # Target zone and the debug marker visibility, should be (F, F).
        self.target_zone_visible = False
        self.target_debug_markers = False

    def reset(self, env, last_info=None):
        """Reset to start an episode.

        If generating training data for goal-conditioned Transporters with
        `main.py` or goal images using `generate_goals.py`, then call the
        superclass. The code already puts the bead poses inside `info`. For
        this env it's IDs 4 through 27 (for 24 beads) but I scale it based on
        num_parts in case we change this value.

        If loading using `load.py` (detect with self.goal_cond_testing) then
        must make targets based on loaded info. However, we still have to
        randomly create the cable, so the easiest way might be to make the
        cable as usual, and then just override the 'places' key later.
        """
        super().reset(env)

        # Override with targets from last_info (and the goal image!).
        if self.goal_cond_testing:
            assert last_info is not None
            self.goal['places'] = self._get_goal_info(last_info)

    def _get_goal_info(self, last_info):
        """Used to determine the goal given the last `info` dict."""
        start_ID = 4
        end_ID = start_ID + self.num_parts
        places = {}
        for ID in range(start_ID, end_ID):
            assert ID in last_info, f'something went wrong with ID={ID}'
            position, _, _ = last_info[ID]
            places[ID] = (position, (0, 0, 0, 1.))
        return places


class CableLineNoTarget(CableShape):
    """Like CableShapeNoTarget, but only straight lines (no visible targets)."""

    def __init__(self):
        super().__init__()
        self._name = 'cable-line-notarget'
        self._debug = False

        # Major change, only considering straight lines.
        self.num_sides_low = 1
        self.num_sides_high = 1

        # Target zone and the debug marker visibility, should be (F, F).
        self.target_zone_visible = False
        self.target_debug_markers = False

    def reset(self, env, last_info=None):
        """See `CableShapeNoTarget.reset()`."""
        super().reset(env)

        # Override with targets from last_info (and the goal image!).
        if self.goal_cond_testing:
            assert last_info is not None
            self.goal['places'] = self._get_goal_info(last_info)

    def _get_goal_info(self, last_info):
        """See `CableShapeNoTarget._get_goal_info()`."""
        start_ID = 4
        end_ID = start_ID + self.num_parts
        places = {}
        for ID in range(start_ID, end_ID):
            assert ID in last_info, f'something went wrong with ID={ID}'
            position, _, _ = last_info[ID]
            places[ID] = (position, (0, 0, 0, 1.))
        return places


class CableRing(CableEnv):
    """Cable as a ring, and no other items.

    This differs from CableShape in that (1) the cable is a ring and
    continuously connected, and (2) the target is actually a ring.

    We need good parameters for num_parts, radius, and ring_radius. So far I
    like these combinations: (24, 0.005, 0.06), (32, 0.005, 0.075), (36,
    0.005, 0.09)... I think using 32 parts to the bead is ideal, given that
    it's the same number as what the bag uses. Also, I increased the postpick
    and preplace z values a bit to let the cables go above each other, to
    avoid the demonstrator going back-and-forth.

    After doing some testing, I think that we might actually want the same
    reward as the bag-alone-open.
    """

    def __init__(self):
        super().__init__()
        self.metric = 'cable-ring'
        self.max_steps = 21
        self.primitive = 'pick_place'
        self._name = 'cable-ring'
        self._debug = False

        # Cable parameters. We use ring_radius to determine sampling bounds.
        self.num_parts = 32
        self.radius = 0.005
        self.ring_radius = 0.075
        self.targets_visible = True

        # Parameters for pick_place primitive.
        self.primitive_params = {
            1: {'speed': 0.001,
                'delta_z': -0.001,
                'postpick_z': 0.04,
                'preplace_z': 0.04,
                'pause_place': 0.0,
            },
        }
        self.task_stage = 1

    def reset(self, env):
        self.total_rewards = 0
        self.object_points = {}
        self.t = 0
        self.task_stage = 1
        self.cable_bead_IDs = []

        # We need this to use the built-in pick_place demonstrator in `task.py`.
        self.goal = {'places': {}, 'steps': [{}]}

        # Sample the center of the ring, increasing size to allow for random force.
        boundary_size = (self.ring_radius * 3, self.ring_radius * 3, 0)
        boundary_pose = self.random_pose(env, boundary_size)
        self.zone_pose = (boundary_pose[0], (0,0,0,1))

        # Add cable ring.
        info = {'center_position': self.zone_pose[0],
                'ring_radius': self.ring_radius,
                'targets_visible': self.targets_visible,}
        self.add_cable_ring(env, info=info)

        # Env must begin before we can apply forces.
        if self._debug:
            self.debug()
        env.start()

        # Add a small force to perturb the cable. Pick a bead at random.
        bead_idx = np.random.randint(len(self.cable_bead_IDs))
        bead_id = self.cable_bead_IDs[bead_idx]
        fx = np.random.randint(low=-20, high=20+1)
        fy = np.random.randint(low=-20, high=20+1)
        fz = 40
        for _ in range(20):
            p.applyExternalForce(bead_id, linkIndex=-1, forceObj=[fx, fy, fz],
                        posObj=[0,0,0], flags=p.LINK_FRAME)
            if self._debug:
                print(f'Perturbing {bead_id}: [{fx:0.2f}, {fy:0.2f}, {fz:0.2f}]')

        time.sleep(self._settle_secs)
        env.pause()


class CableRingNoTarget(CableRing):
    """Cable as a ring, but no target, so it subclasses CableRing."""

    def __init__(self):
        super().__init__()
        self._name = 'cable-ring-notarget'
        self._debug = False
        self.targets_visible = False

    def reset(self, env):
        super().reset(env)
