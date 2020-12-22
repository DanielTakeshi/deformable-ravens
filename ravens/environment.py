#!/usr/bin/env python

import os
import sys
import time
import threading
import pkg_resources

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from ravens.gripper import Gripper, Suction
from ravens import tasks, utils


class Environment():

    def __init__(self, disp=False, hz=240):
        """Creates OpenAI gym-style env with support for PyBullet threading.

        Args:
            disp: Whether or not to use PyBullet's built-in display viewer.
                Use this either for local inspection of PyBullet, or when
                using any soft body (cloth or bags), because PyBullet's
                TinyRenderer graphics (used if disp=False) will make soft
                bodies invisible.
            hz: Parameter used in PyBullet to control the number of physics
                simulation steps. Higher values lead to more accurate physics
                at the cost of slower computaiton time. By default, PyBullet
                uses 240, but for soft bodies we need this to be at least 480
                to avoid cloth intersecting with the plane.
        """
        self.ee = None
        self.task = None
        self.objects = []
        self.running = False
        self.fixed_objects = []
        self.pix_size = 0.003125
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.primitives = {'push':       self.push,
                           'sweep':      self.sweep,
                           'pick_place': self.pick_place}

        # Set default movej timeout limit. For most tasks, 15 is reasonable.
        self.t_lim = 15

        # From Xuchen: need this for using any new deformable simulation.
        self.use_new_deformable = True
        self.hz = hz

        # Start PyBullet.
        p.connect(p.GUI if disp else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        assets_path = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(assets_path)

        # Check PyBullet version (see also the cloth/bag task scripts!).
        p_version = pkg_resources.get_distribution('pybullet').version
        tested = ['2.8.4', '3.0.4']
        assert p_version in tested, f'PyBullet version {p_version} not in {tested}'

        # Move the camera a little closer to the scene. Most args are not used.
        # PyBullet defaults: yaw=50 and pitch=-35.
        if disp:
            _, _, _, _, _, _, _, _, _, _, _, target = p.getDebugVisualizerCamera()
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target,)

        # Control PyBullet simulation steps.
        self.step_thread = threading.Thread(target=self.step_simulation)
        self.step_thread.daemon = True
        self.step_thread.start()

    def step_simulation(self):
        """Adding optional hertz parameter for better cloth physics.

        From our discussion with Erwin, we should just set time.sleep(0.001),
        or even consider removing it all together. It's mainly for us to
        visualize PyBullet with the GUI to make it not move too fast
        """
        p.setTimeStep(1.0 / self.hz)
        while True:
            if self.running:
                p.stepSimulation()
            if self.ee is not None:
                self.ee.step()
            time.sleep(0.001)

    def stop(self):
        p.disconnect()
        del self.step_thread

    def start(self):
        self.running = True

    def pause(self):
        self.running = False

    def is_static(self):
        """Checks if env is static, used for checking if action finished.

        However, this won't work in PyBullet (at least v2.8.4) since soft
        bodies cause this code to hang. Therefore, look at the task's
        `def_IDs` list, which by design will have all IDs of soft bodies.
        Furthermore, for the bag tasks, the beads generally move around, so
        for those, just use a hard cutoff limit (outside this method).
        """
        if self.is_softbody_env():
            assert len(self.task.def_IDs) > 0, 'Did we forget to add to def_IDs?'
            v = [np.linalg.norm(p.getBaseVelocity(i)[0]) for i in self.objects
                    if i not in self.task.def_IDs]
        else:
            v = [np.linalg.norm(p.getBaseVelocity(i)[0]) for i in self.objects]
        return all(np.array(v) < 1e-2)

    def add_object(self, urdf, pose, fixed=False):
        fixedBase = 1 if fixed else 0
        object_id = p.loadURDF(urdf, pose[0], pose[1], useFixedBase=fixedBase)
        if fixed:
            self.fixed_objects.append(object_id)
        else:
            self.objects.append(object_id)
        return object_id

    #-------------------------------------------------------------------------
    # Standard Gym Functions
    #-------------------------------------------------------------------------

    def reset(self, task, last_info=None, disable_render_load=True):
        """Sets up PyBullet, loads models, resets the specific task.

        We do a step() call with act=None at the end. This will only return
        an empty obs dict, obs={}. For some tasks where the reward could be
        nonzero at the start, we can report the reward shown here.

        Args:
            last_info: Only for goal-conditioned learning DURING TEST TIME,
                since we load in a target image, but we also want to load in
                final object poses, since in many cases we get better
                accuracy. For simplicity, I suggest we put all this in the
                `info` dict, and we load it as the `info` that happens after
                finishing. That will have the most up to date object poses,
                and we can always add extra information as needed in
                last_info['extras'].
            disable_render_load: Need this as True to avoid `p.loadURDF`
                becoming a time bottleneck, judging from my profiling.
        """
        self.pause()
        self.task = task
        self.objects = []
        self.fixed_objects = []
        if self.use_new_deformable:
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        else:
            p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Slightly increase default movej timeout for the more demanding tasks.
        if self.is_bag_env():
            self.t_lim = 60
            if isinstance(self.task, tasks.names['bag-color-goal']):
                self.t_lim = 120

        # Empirically, this seems to make loading URDFs faster w/remote displays.
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        id_plane = p.loadURDF('assets/plane/plane.urdf', [0, 0, -0.001])
        id_ws = p.loadURDF('assets/ur5/workspace.urdf', [0.5, 0, 0])

        # Load UR5 robot arm equipped with task-specific end effector.
        self.ur5 = p.loadURDF(f'assets/ur5/ur5-{self.task.ee}.urdf')
        self.ee_tip_link = 12
        if self.task.ee == 'suction':
            self.ee = Suction(self.ur5, 11)
        elif self.task.ee == 'gripper':
            self.ee = Robotiq2F85(self.ur5, 9)
            self.ee_tip_link = 10
        else:
            self.ee = Gripper()

        # Get revolute joint indices of robot (skip fixed joints).
        num_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(num_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        # Get end effector tip pose in home configuration.
        ee_tip_state = p.getLinkState(self.ur5, self.ee_tip_link)
        self.home_pose = np.array(ee_tip_state[0] + ee_tip_state[1])

        # Reset end effector.
        self.ee.release()

        # Seems like this should be BEFORE reset()
        # since for bag-items we may assign to True!
        task.exit_gracefully = False

        # Reset task.
        if last_info is not None:
            task.reset(self, last_info)
        else:
            task.reset(self)

        # Daniel: might be useful to have this debugging tracker.
        self.IDTracker = utils.TrackIDs()
        self.IDTracker.add(id_plane, 'Plane')
        self.IDTracker.add(id_ws, 'Workspace')
        self.IDTracker.add(self.ur5, 'UR5')
        try:
            self.IDTracker.add(self.ee.body, 'Gripper.body')
        except:
            pass

        # Daniel: add other IDs, but not all envs use the ID tracker.
        try:
            task_IDs = task.get_ID_tracker()
            for i in task_IDs:
                self.IDTracker.add(i, task_IDs[i])
        except AttributeError:
            pass
        #print(self.IDTracker)  # If doing multiple episodes, check if I reset the ID dict!
        assert id_ws == 1, f'Workspace ID: {id_ws}'

        # Daniel: tune gripper for deformables if applicable, and CHECK HZ!!
        if self.is_softbody_env():
            self.ee.set_def_threshold(threshold=self.task.def_threshold)
            self.ee.set_def_nb_anchors(nb_anchors=self.task.def_nb_anchors)
            assert self.hz >= 480, f'Error, hz={self.hz} is too small!'

        # Restart simulation.
        self.start()
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        (obs, _, _, _) = self.step()
        return obs

    def step(self, act=None):
        """Execute action with specified primitive.

        For each episode (training, loading, etc.), this is normally called the first
        time from `env.reset()` above, with NO action, and returns an EMPTY observation.
        Then, it's called a SECOND time with an action that lacks a primitive (set to
        None, even though the key exists). But, this method will return an actual image
        observation that we pass to the policy. Finally, subsequent calls will have
        proper actions.

        (Sept 08) Added graceful exit functionality. This will let the code terminate
        early but with task.done=False as we didn't actually 'finish' -- just failed.
        Should use task.done=True to indicate any form of 'successful' dones (hitting
        time limit doesn't count).

        (Oct 09) Clarify documentation for the confusing first time step. I now see
        with ground truth agents, there is no 'second action lacking a primitive',
        because ground truth agents don't need images (see their `act` method).
        """
        if act and act['primitive']:
            success = self.primitives[act['primitive']](**act['params'])

            # Exit early if action failed. Daniel: adding exit_gracefully.
            if (not success) or self.task.exit_gracefully:
                _, reward_extras = self.task.reward()
                info = self.info
                reward_extras['task.done'] = False

                # Means we hit irrecoverable action, exit now (reset to False!!).
                if self.task.exit_gracefully:
                    reward_extras['exit_gracefully'] = True
                    self.task.exit_gracefully = False  # important !!!

                # For consistency?
                if isinstance(self.task, tasks.names['cloth-flat-notarget']):
                    info['sampled_zone_pose'] = self.task.zone_pose
                elif isinstance(self.task, tasks.names['bag-color-goal']):
                    info['bag_base_pos'] = self.task.bag_base_pos[0]
                    info['bag_base_orn'] = self.task.bag_base_orn[0]
                    info['bag_target_color'] = self.task.bag_colors[0]

                info['extras'] = reward_extras
                return {}, 0, True, info

        # Wait for objects to settle, with a hard exit for bag tasks.
        start_t = time.time()
        while not self.is_static():
            if self.is_bag_env() and (time.time() - start_t > 2.0):
                break
            time.sleep(0.001)

        # Compute task rewards.
        reward, reward_extras = self.task.reward()
        done = self.task.done()

        # Pass ground truth robot state as info.
        info = self.info

        # Daniel: fine-grained info about rewards (since it's nuanced for some tasks).
        # If we hit time limit, `task.done` will check if we succeeded on last action.
        reward_extras['task.done'] = done
        info['extras'] = reward_extras
        if isinstance(self.task, tasks.names['cloth-flat-notarget']):
            info['sampled_zone_pose'] = self.task.zone_pose
        elif isinstance(self.task, tasks.names['bag-color-goal']):
            info['bag_base_pos'] = self.task.bag_base_pos[0]
            info['bag_base_orn'] = self.task.bag_base_orn[0]
            info['bag_target_color'] = self.task.bag_colors[0]

        # Get camera observations per specified config.
        obs = {}
        if act and 'camera_config' in act:
            obs['color'], obs['depth'] = [], []
            for config in act['camera_config']:
                color, depth, _ = self.render(config)
                obs['color'].append(color)
                obs['depth'].append(depth)

        return obs, reward, done, info

    def render(self, config):
        """Render RGB-D image with specified configuration."""

        # Compute OpenGL camera settings.
        lookdir = np.array([0, 0, 1]).reshape(3, 1)
        updir = np.array([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.array(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_length = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (np.arctan((config['image_size'][0] /
                           2) / focal_length) * 2 / np.pi) * 180

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config['image_size'][1],
            height=config['image_size'][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (config['image_size'][0],
                            config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        color_image_size = (color_image_size[0], color_image_size[1], 3)
        if config['noise']:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color_image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += np.random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    @property
    def info(self):
        """Normally returns dict of:

            object id : (position, rotation, dimensions)

        However, I made a few changes. First, removing IDs in some tasks, so
        we shouldn't query their position. Second, adding object mesh for
        gt_state and soft bodies. The second `if` test is redundant but just
        in case. Here, we instead map to the mesh data instead of (position,
        rotation, dimension). This is: (nb_vertices, (positions)) where the
        latter is a tuple that has all the 3D positions of each vertex in the
        simulation mesh, e.g., it's 100 for cloth.

        Note on soft body IDs:
        cloth-cover: ID 5 is cloth (ID 4 is item to cover)
        cloth-flat(notarget): ID 5 is cloth (ID 4 is the zone, though in the no
            target case we remove it ... historical reasons).
        bag tasks: all have ID 5 as the bag, because they use ID 4 for the zone,
            (with bag-alone removing zone) and other items are added afterwards.

        To see how we use the special case for soft bodies, see:
            ravens/agents/gt_state.py and the extraact_x_y_theta method.
        We depend on assuming that len(info[id]) = 2 instead of 3.
        """
        removed_IDs = []
        if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
                isinstance(self.task, tasks.names['bag-alone-open'])):
            removed_IDs.append(self.task.zone_ID)

        # Daniel: special case for soft bodies (and gt_state). For now only cloth.
        softbody_id = -1
        if self.is_cloth_env():
            assert len(self.task.def_IDs) == 1, self.task.def_IDs
            softbody_id = self.task.def_IDs[0]

        # object id : (position, rotation, dimensions)
        info = {}
        for object_id in (self.fixed_objects + self.objects):
            if object_id in removed_IDs:
                continue

            # Daniel: special cases for soft bodies (and gt_state), and then bags.
            if (object_id == softbody_id) and self.is_cloth_env():
                info[object_id] = p.getMeshData(object_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
            elif isinstance(self.task, tasks.names['bag-color-goal']):
                # Note: we don't do this for sorting? :X
                position, rotation = p.getBasePositionAndOrientation(object_id)
                dimensions = p.getVisualShapeData(object_id)[0][3]
                rgba_color = p.getVisualShapeData(object_id)[0][7]  # see pybullet docs
                assert rgba_color[3] == 1, rgba_color
                rgb_color = rgba_color[0:3]  # we can ignore the last component it is always 1
                info[object_id] = (position, rotation, dimensions, rgb_color)
            else:
                # The usual case.
                position, rotation = p.getBasePositionAndOrientation(object_id)
                dimensions = p.getVisualShapeData(object_id)[0][3]
                info[object_id] = (position, rotation, dimensions)

        return info

    #-------------------------------------------------------------------------
    # Robot Movement Functions
    #-------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, t_lim=20):
        """Move UR5 to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < t_lim:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            time.sleep(0.001)
        print('Warning: movej exceeded {} sec timeout. Skipping.'.format(t_lim))
        return False

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        # # Keep joint angles between -180/+180
        # targj[5] = ((targj[5] + np.pi) % (2 * np.pi) - np.pi)
        targj = self.solve_IK(pose)
        return self.movej(targj, speed, self.t_lim)

    def solve_IK(self, pose):
        homej_list = np.array(self.homej).tolist()
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip_link,
            targetPosition=pose[:3],
            targetOrientation=pose[3:],
            lowerLimits=[-17, -2.3562, -17, -17, -17, -17],
            upperLimits=[17, 0, 17, 17, 17, 17],
            jointRanges=[17] * 6,
            restPoses=homej_list,
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.array(joints)
        joints[joints > 2 * np.pi] = joints[joints > 2 * np.pi] - 2 * np.pi
        joints[joints < -2 * np.pi] = joints[joints < -2 * np.pi] + 2 * np.pi
        return joints

    #-------------------------------------------------------------------------
    # Motion Primitives
    #-------------------------------------------------------------------------

    def pick_place(self, pose0, pose1):
        """Execute pick and place primitive.

        Standard ravens tasks use the `delta` vector to lower the gripper
        until it makes contact with something. With deformables, however, we
        need to consider cases when the gripper could detect a rigid OR a
        soft body (cloth or bag); it should grip the first item it touches.
        This is handled in the Gripper class.

        Different deformable ravens tasks use slightly different parameters
        for better physics (and in some cases, faster simulation). Therefore,
        rather than make special cases here, those tasks will define their
        own action parameters, which we use here if they exist. Otherwise, we
        stick to defaults from standard ravens. Possible action parameters a
        task might adjust:

            speed: how fast the gripper moves.
            delta_z: how fast the gripper lowers for picking / placing.
            prepick_z: height of the gripper when it goes above the target
                pose for picking, just before it lowers.
            postpick_z: after suction gripping, raise to this height, should
                generally be low for cables / cloth.
            preplace_z: like prepick_z, but for the placing pose.
            pause_place: add a small pause for some tasks (e.g., bags) for
                slightly better soft body physics.
            final_z: height of the gripper after the action. Recommended to
                leave it at the default of 0.3, because it has to be set high
                enough to avoid the gripper occluding the workspace when
                generating color/depth maps.
        Args:
            pose0: picking pose.
            pose1: placing pose.

        Returns:
            A bool indicating whether the action succeeded or not, via
            checking the sequence of movep calls. If any movep failed, then
            self.step() will terminate the episode after this action.
        """
        # Defaults used in the standard Ravens environments.
        speed = 0.01
        delta_z = -0.001
        prepick_z = 0.3
        postpick_z = 0.3
        preplace_z = 0.3
        pause_place = 0.0
        final_z = 0.3

        # Find parameters, which may depend on the task stage.
        if hasattr(self.task, 'primitive_params'):
            ts = self.task.task_stage
            if 'prepick_z' in self.task.primitive_params[ts]:
                prepick_z = self.task.primitive_params[ts]['prepick_z']
            speed       = self.task.primitive_params[ts]['speed']
            delta_z     = self.task.primitive_params[ts]['delta_z']
            postpick_z  = self.task.primitive_params[ts]['postpick_z']
            preplace_z  = self.task.primitive_params[ts]['preplace_z']
            pause_place = self.task.primitive_params[ts]['pause_place']

        # Used to track deformable IDs, so that we can get the vertices.
        def_IDs = []
        if hasattr(self.task, 'def_IDs'):
            def_IDs = self.task.def_IDs

        # Otherwise, proceed as normal.
        success = True
        pick_position = np.array(pose0[0])
        pick_rotation = np.array(pose0[1])
        prepick_position = pick_position.copy()
        prepick_position[2] = prepick_z

        # Execute picking motion primitive.
        prepick_pose = np.hstack((prepick_position, pick_rotation))
        success &= self.movep(prepick_pose)
        target_pose = prepick_pose.copy()
        delta = np.array([0, 0, delta_z, 0, 0, 0, 0])

        # Lower gripper until (a) touch object (rigid OR softbody), or (b) hit ground.
        while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
            target_pose += delta
            success &= self.movep(target_pose)

        # Create constraint (rigid objects) or anchor (deformable).
        self.ee.activate(self.objects, def_IDs)

        # Increase z slightly (or hard-code it) and check picking success.
        if self.is_softbody_env() or self.is_new_cable_env():
            prepick_pose[2] = postpick_z
            success &= self.movep(prepick_pose, speed=speed)
            time.sleep(pause_place) # extra rest for bags
        elif isinstance(self.task, tasks.names['cable']):
            prepick_pose[2] = 0.03
            success &= self.movep(prepick_pose, speed=0.001)
        else:
            prepick_pose[2] += pick_position[2]
            success &= self.movep(prepick_pose)
        pick_success = self.ee.check_grasp()

        if pick_success:
            place_position = np.array(pose1[0])
            place_rotation = np.array(pose1[1])
            preplace_position = place_position.copy()
            preplace_position[2] = 0.3 + pick_position[2]

            # Execute placing motion primitive if pick success.
            preplace_pose = np.hstack((preplace_position, place_rotation))
            if self.is_softbody_env() or self.is_new_cable_env():
                preplace_pose[2] = preplace_z
                success &= self.movep(preplace_pose, speed=speed)
                time.sleep(pause_place) # extra rest for bags
            elif isinstance(self.task, tasks.names['cable']):
                preplace_pose[2] = 0.03
                success &= self.movep(preplace_pose, speed=0.001)
            else:
                success &= self.movep(preplace_pose)

            # Lower the gripper. Here, we have a fixed speed=0.01. TODO: consider additional
            # testing with bags, so that the 'lowering' process for bags is more reliable.
            target_pose = preplace_pose.copy()
            while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
                target_pose += delta
                success &= self.movep(target_pose)

            # Release AND get gripper high up, to clear the view for images.
            self.ee.release()
            preplace_pose[2] = final_z
            success &= self.movep(preplace_pose)
        else:
            # Release AND get gripper high up, to clear the view for images.
            self.ee.release()
            prepick_pose[2] = final_z
            success &= self.movep(prepick_pose)
        return success

    def sweep(self, pose0, pose1):
        """Execute sweeping primitive."""
        success = True
        position0 = np.float32(pose0[0])
        position1 = np.float32(pose1[0])
        direction = position1 - position0
        length = np.linalg.norm(position1 - position0)
        if length == 0:
            direction = np.float32([0, 0, 0])
        else:
            direction = (position1 - position0) / length

        theta = np.arctan2(direction[1], direction[0])
        rotation = p.getQuaternionFromEuler((0, 0, theta))

        over0 = position0.copy()
        over0[2] = 0.3
        over1 = position1.copy()
        over1[2] = 0.3

        success &= self.movep(np.hstack((over0, rotation)))
        success &= self.movep(np.hstack((position0, rotation)))

        num_pushes = np.int32(np.floor(length / 0.01))
        for i in range(num_pushes):
            target = position0 + direction * num_pushes * 0.01
            success &= self.movep(np.hstack((target, rotation)), speed=0.003)

        success &= self.movep(np.hstack((position1, rotation)), speed=0.003)
        success &= self.movep(np.hstack((over1, rotation)))
        return success

    def push(self, pose0, pose1):
        """Execute pushing primitive."""
        p0 = np.float32(pose0[0])
        p1 = np.float32(pose1[0])
        p0[2], p1[2] = 0.025, 0.025
        if np.sum(p1 - p0) == 0:
            push_direction = 0
        else:
            push_direction = (p1 - p0) / np.linalg.norm((p1 - p0))
        p1 = p0 + push_direction * 0.01
        success &= self.movep(np.hstack((p0, self.home_pose[3:])))
        success &= self.movep(np.hstack((p1, self.home_pose[3:])), speed=0.003)
        return success

    #-------------------------------------------------------------------------
    # Motion Primitives
    #-------------------------------------------------------------------------

    def is_softbody_env(self):
        """In addition to this, please check task.py. In particular...

        Check the (a) policy's action, (b) reward, (c) done conditions.
        """
        return self.is_cloth_env() or self.is_bag_env()

    def is_new_cable_env(self):
        """I want a way to track new cable-related stuff alone."""
        return (isinstance(self.task, tasks.names['cable-shape']) or
                isinstance(self.task, tasks.names['cable-shape-notarget']) or
                isinstance(self.task, tasks.names['cable-line-notarget']) or
                isinstance(self.task, tasks.names['cable-ring']) or
                isinstance(self.task, tasks.names['cable-ring-notarget']))

    def is_cloth_env(self):
        """Keep this updated when I adjust environment names."""
        return (isinstance(self.task, tasks.names['cloth-flat']) or
                isinstance(self.task, tasks.names['cloth-flat-notarget']) or
                isinstance(self.task, tasks.names['cloth-cover']))

    def is_bag_env(self):
        """Keep this updated when I adjust environment names."""
        return (isinstance(self.task, tasks.names['bag-alone-open']) or
                isinstance(self.task, tasks.names['bag-items-easy']) or
                isinstance(self.task, tasks.names['bag-items-hard']) or
                isinstance(self.task, tasks.names['bag-color-goal']))
