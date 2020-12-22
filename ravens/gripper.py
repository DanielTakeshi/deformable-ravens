#!/usr/bin/env python

import time
import threading
import numpy as np
import pybullet as p
from ravens import utils as U


class Gripper:

    def __init__(self):
        self.activated = False

    def step(self):
        return

    def activate(self, objects):
        return

    def release(self):
        return

#-----------------------------------------------------------------------------
# Suction-Based Gripper
#-----------------------------------------------------------------------------

class Suction(Gripper):

    def __init__(self, robot_id, tool_link):
        """Creates constraint between suction and the robot.

        p.getNumJoints(robot_id) = 14
        p.getNumJoints(self.body) = 1

        Constraint ID is 1 (if this is the first constraint) and parent and
        child object IDs are 2 (UR5) and 3 (Gripper), respectively.

        Has special cases when dealing with rigid vs deformables. For rigid,
        only need to check contact_constraint for any constraint. For cloth,
        use cloth_threshold to check distances from gripper body (self.body)
        to any vertex in the cloth mesh.

        To get the suction gripper pose, use p.getLinkState(self.body, 0),
        and not p.getBasePositionAndOrientation(self.body) as the latter is
        about z=0.03m higher and empirically seems worse.
        """
        position = (0.487, 0.109, 0.351)
        rotation = p.getQuaternionFromEuler((np.pi, 0, 0))
        urdf = 'assets/ur5/suction/suction-head.urdf'
        self.body = p.loadURDF(urdf, position, rotation)
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=tool_link,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.07))
        p.changeConstraint(constraint_id, maxForce=50)
        self.activated = False
        self.contact_constraint = None

        # Default values for deformables, but can override in environment.py.
        # Suction must be within `def_ignore` distance from a vertex to grip a
        # deformable -- otherwise just ignore it. (Needs to be tuned.) Update:
        # with the recent change to supprt deformables and rigid, we can only
        # grip a deformable we made contact with, so the `def_ignore` threshold
        # shouldn't be applied ... unless perhaps distances change between when
        # contact was detected and when activate is called?
        self.def_ignore = 0.035
        self.def_threshold = 0.030
        self.def_nb_anchors = 1

        # Update Oct 22: argh, I now see from bag testing that sometimes
        # when we move the block in midair, this distance can go below 0.94. :(
        # Let's try 0.93 and hope this prevents releasing in midair.
        self.def_frac_lower = 0.93
        self.def_frac_upper = 2.00

        # Track which deformable is being gripped, and anchors.
        self.def_grip_item = None
        self.def_grip_anchors = None

        # Gripping a deformable object which touches a rigid OR deformable.
        self.def_min_vetex = None
        self.def_min_distance = None

        # Gripping a rigid object which touches a deformable.
        self.init_grip_distance = None
        self.init_grip_item = None

    def activate(self, possible_objects, def_IDs):
        """
        Simulates suction by creating rigid fixed constraint between suction
        gripper and contacted object.

        :def_IDs: a list of IDs of deformable objects.
        """
        if not self.activated:
            # Only report contact points involving linkIndexA of bodyA (the
            # suction) -- returns a list (actually, a tuple) of such points.
            points = p.getContactPoints(bodyA=self.body, linkIndexA=0)

            if len(points) > 0:
                # Handle contact with a rigid object.
                for point in points:
                    object_id, contact_link = point[2], point[4]
                if object_id in possible_objects:
                    body_pose = p.getLinkState(self.body, 0)
                    object_pose = p.getBasePositionAndOrientation(object_id)
                    world_to_body = p.invertTransform(
                        body_pose[0], body_pose[1])
                    object_to_body = p.multiplyTransforms(
                        world_to_body[0], world_to_body[1],
                        object_pose[0], object_pose[1])
                    self.contact_constraint = p.createConstraint(
                        parentBodyUniqueId=self.body,
                        parentLinkIndex=0,
                        childBodyUniqueId=object_id,
                        childLinkIndex=contact_link,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=object_to_body[0],
                        parentFrameOrientation=object_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0))
                    # Handle the case when rigid item makes contact with a
                    # deformable, which will cause this distance to shrink.
                    # Assumes gripper is suctioning ONE rigid item at a time.
                    distance = np.linalg.norm(
                            np.float32(body_pose[0]) - np.float32(object_pose[0]))
                    self.init_grip_distance = distance
                    self.init_grip_item = object_id
                #print(f'Gripping a rigid item!')
            elif (self.def_grip_item is not None):
                # Otherwise, focus on gripping a _deformable_ with anchors.
                info = self.activate_def(self.def_grip_item)
                self.def_grip_anchors = info['anchors']
                self.def_min_vertex = info['closest_vertex']
                self.def_min_distance = info['min_distance']
                #print(f'Gripping a deformable!')

            self.activated = True

    def activate_def(self, defId):
        """Simulates suction by anchoring vertices of the deformable object.

        Get distance values in `distances`, get indices for argsort, then
        resulting indices in `distances_sort` correspond _exactly_ to vertex
        indices arranged from nearest to furthest to the gripper.
        """
        _, vert_pos_l = p.getMeshData(defId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        gripper_position = np.float32(p.getLinkState(self.body, 0)[0])
        distances = []
        for v_position in vert_pos_l:
            d = gripper_position - np.float32(v_position)
            distances.append(np.linalg.norm(d))
        distances_sort = np.argsort(distances)

        anchors = []
        for i in range(self.def_nb_anchors):
            # For each vertex close enough (under threshold), create anchor(s).
            vIndex = distances_sort[i]
            if distances[vIndex] > self.def_threshold:
                #print(f'WARNING, dist={distances[vIndex]:0.4f} > {self.def_threshold} '
                #    f'This means our gripper touched the surface (z=0)')
                pass
            # This should prevent us from gripping if the suction didn't grip anything.
            if distances[vIndex] > self.def_ignore:
                print(f'WARNING, dist={distances[vIndex]:0.4f} > thresh '
                    f'{self.def_ignore:0.4f}. No points are close to the suction')
                break
            anchorId = p.createSoftBodyAnchor(
                    softBodyBodyUniqueId=defId,
                    nodeIndex=vIndex,
                    bodyUniqueId=self.body,
                    linkIndex=-1,)
            anchors.append(anchorId)

        info = {'anchors': anchors,
                'closest_vertex': distances_sort[0],
                'min_distance': np.min(distances),}
        #print(info)
        return info

    def release(self):
        """
        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.
        """
        if self.activated:
            self.activated = False

            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:
                    pass
                self.init_grip_distance = None
                self.init_grip_item = None

            # Release gripped deformable object (if any).
            if self.def_grip_anchors is not None:
                for anchorId in self.def_grip_anchors:
                    p.removeConstraint(anchorId)
                self.def_grip_anchors = None
                self.def_grip_item = None
                self.def_min_vetex = None
                self.def_min_distance = None

    def detect_contact(self, def_IDs):
        """
        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.

        :def_IDs: a list of IDs of deformable objects. Since it may be
        computationally heavy to check, if we have any contact points with
        rigid we just return that right away (w/out checking deformables).
        """
        body, link = self.body, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:
                self.contact_constraint = None
                pass

        # Get all contact points between suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
            points = [point for point in points if point[2] != self.body]

        # Normally we return if len(points) > 0, but now if len == 0, we might be
        # missing (a) gripping a deformable or (b) rigid item hitting a deformable.
        if len(points) > 0:
            return True

        # If suction off and len(points)==0, check contact w/deformables. Note:
        # it is critical that we set `self.def_grip_item` correctly.
        if not self.activated:
            gripper_position = np.float32(p.getLinkState(self.body, 0)[0])
            for ID in def_IDs:
                if self.detect_contact_def(gripper_position, ID):
                    self.def_grip_item = ID
                    return True
            return False

        # Here, we're either (a) gripping a deformable, or (b) gripping a rigid
        # item that is NOT touching another rigid item.
        assert self.activated

        if (self.init_grip_item is not None):
            # If suction on, check if a gripped RIGID item touches a deformable.
            # For this, tests were applying if the gripper went through the
            # rigid item, but perhaps also if the fraction is too high?
            # [Haven't seen this in any test scenario, though.]
            object_pose = p.getBasePositionAndOrientation(self.init_grip_item)
            gripper_position = np.float32(p.getLinkState(self.body, 0)[0])
            d = gripper_position - np.float32(object_pose[0])
            distance = np.linalg.norm(d)
            fraction = distance / self.init_grip_distance
            if (fraction <= self.def_frac_lower) or (fraction >= self.def_frac_upper):
                #print(f'[gripping rigid], distance: {distance:0.5f} vs original: '
                #    f'{self.init_grip_distance:0.5f}, and fraction: {fraction:0.5f}')
                # Will release this item, so we don't need these until the next suction.
                self.init_grip_distance = None
                self.init_grip_item = None
            return (fraction <= self.def_frac_lower) or (fraction >= self.def_frac_upper)

        elif (self.def_grip_item is not None):
            # Should see if I ever encounter this in practice? With cloth-cover
            # and cloth-flat I don't think we trigger this condition. UPDATE:
            # ah, upon further tests, that's because the len(points)>0 trigger
            # kicks in when we grip a deformable, but where the suction makes
            # contact. If that happens I'm fine, reduces complexity.
            #
            # TODO: actually need to fix, I think we need suction gripper and
            # the robot's joint just before that. This will not get activated.
            return False
        else:
            # I don't think this should ever invoke -- we should always be
            # gripping a rigid or deformable in this condition.
            return False

    def detect_contact_def(self, gripper_position, defId):
        """Detect contact, when dealing with deformables.

        We may want to speed this up if it is a bottleneck. Returns a binary
        signal of whether there exists _any_ vertex within the threshold.
        Note with collisionMargin=0.004, I am getting most cloth vertices
        (for ClothFlat) to settle at ~0.004m high.
        """
        _, vert_pos_l = p.getMeshData(defId, -1, flags=p.MESH_DATA_SIMULATION_MESH)

        # Vectorized.
        distances_np = gripper_position - np.array(vert_pos_l)
        assert len(distances_np.shape) == 2, distances_np.shape
        distances_L2 = np.linalg.norm(distances_np, axis=1)
        return np.min(distances_L2) < self.def_threshold

        # Older way
        #distances = []
        #for v_position in vert_pos_l:
        #    d = gripper_position - np.float32(v_position)
        #    distances.append(np.linalg.norm(d))
        #return np.min(distances) < self.def_threshold

    def check_grasp(self):
        """Check a grasp for picking success.

        If picking fails, then robot doesn't do the place action. For rigid
        items: index 2 in getConstraintInfo returns childBodyUniqueId. For
        deformables, check the length of the anchors.
        """
        pick_deformable = False
        if self.def_grip_anchors is not None:
            pick_deformable = len(self.def_grip_anchors) > 0
        return (not self.contact_constraint is None) or pick_deformable

    def set_def_threshold(self, threshold):
        self.def_threshold = threshold

    def set_def_nb_anchors(self, nb_anchors):
        self.def_nb_anchors = nb_anchors

#-----------------------------------------------------------------------------
# Parallel-Jaw Two-Finger Gripper (TODO: fix)
#-----------------------------------------------------------------------------

# class Robotiq2F85:

#     def __init__(self, robot, tool):
#         self.robot = robot
#         self.tool = tool
#         pos = [0.487, 0.109, 0.421]
#         rot = p.getQuaternionFromEuler([np.pi, 0, 0])
#         urdf = 'assets/ur5/gripper/robotiq_2f_85.urdf'
#         self.body = p.loadURDF(urdf, pos, rot)
#         self.n_joints = p.getNumJoints(self.body)
#         self.activated = False

#         # Connect gripper base to robot tool
#         p.createConstraint(self.robot, tool, self.body, 0,
#                            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
#                            parentFramePosition=[0, 0, 0],
#                            childFramePosition=[0, 0, -0.05])

#         # Set friction coefficients for gripper fingers
#         for i in range(p.getNumJoints(self.body)):
#             p.changeDynamics(self.body, i,
#                              lateralFriction=1.5,
#                              spinningFriction=1.0,
#                              rollingFriction=0.0001,
#                              # rollingFriction=1.0,
#                              frictionAnchor=True)  # contactStiffness=0.0, contactDamping=0.0

#         # Start thread to handle additional gripper constraints
#         self.motor_joint = 1
#         # self.constraints_thread = threading.Thread(target=self.step)
#         # self.constraints_thread.daemon = True
#         # self.constraints_thread.start()

#     # Control joint positions by enforcing hard contraints on gripper behavior
#     # Set one joint as the open/close motor joint (other joints should mimic)
#     def step(self):
#         # while True:
#         currj = [p.getJointState(self.body, i)[0]
#                  for i in range(self.n_joints)]
#         indj = [6, 3, 8, 5, 10]
#         targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
#         p.setJointMotorControlArray(self.body, indj, p.POSITION_CONTROL,
#                                     targj, positionGains=np.ones(5))
#         # time.sleep(0.001)

#     # Close gripper fingers and check grasp success (width between fingers
#     # exceeds some threshold)
#     def activate(self, valid_obj=None):
#         p.setJointMotorControl2(self.body, self.motor_joint,
#                                 p.VELOCITY_CONTROL, targetVelocity=1, force=100)
#         if not self.external_contact():
#             while self.moving():
#                 time.sleep(0.001)
#         self.activated = True

#     # Open gripper fingers
#     def release(self):
#         p.setJointMotorControl2(self.body, self.motor_joint,
#                                 p.VELOCITY_CONTROL, targetVelocity=-1, force=100)
#         while self.moving():
#             time.sleep(0.001)
#         self.activated = False

#     # If activated and object in gripper: check object contact
#     # If activated and nothing in gripper: check gripper contact
#     # If released: check proximity to surface
#     def detect_contact(self):
#         obj, link, ray_frac = self.check_proximity()
#         if self.activated:
#             empty = self.grasp_width() < 0.01
#             cbody = self.body if empty else obj
#             if obj == self.body or obj == 0:
#                 return False
#             return self.external_contact(cbody)
#         else:
#             return ray_frac < 0.14 or self.external_contact()

#     # Return if body is in contact with something other than gripper
#     def external_contact(self, body=None):
#         if body is None:
#             body = self.body
#         pts = p.getContactPoints(bodyA=body)
#         pts = [pt for pt in pts if pt[2] != self.body]
#         return len(pts) > 0

#     # Check grasp success
#     def check_grasp(self):
#         while self.moving():
#             time.sleep(0.001)
#         success = self.grasp_width() > 0.01
#         return success

#     def grasp_width(self):
#         lpad = np.array(p.getLinkState(self.body, 4)[0])
#         rpad = np.array(p.getLinkState(self.body, 9)[0])
#         dist = np.linalg.norm(lpad - rpad) - 0.047813
#         return dist

#     # Helper functions

#     def moving(self):
#         v = [np.linalg.norm(p.getLinkState(
#             self.body, i, computeLinkVelocity=1)[6]) for i in [3, 8]]
#         return any(np.array(v) > 1e-2)

#     def check_proximity(self):
#         ee_pos = np.array(p.getLinkState(self.robot, self.tool)[0])
#         tool_pos = np.array(p.getLinkState(self.body, 0)[0])
#         vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
#         ee_targ = ee_pos + vec
#         ray_data = p.rayTest(ee_pos, ee_targ)[0]
#         obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
#         return obj, link, ray_frac
