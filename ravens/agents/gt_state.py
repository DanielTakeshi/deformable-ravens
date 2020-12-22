#!/usr/bin/env python
import os
import cv2
import pickle
import numpy as np
import pybullet as p
import time
import transformations
import matplotlib.pyplot as plt
import tensorflow as tf; tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from ravens import cameras
from ravens import utils
from ravens.models import MlpModel
from ravens.models import mdn_utils

TASKS_SOFT = ['cloth-cover', 'cloth-flat', 'cloth-flat-notarget']


class GtStateAgent:
    """Agent which uses ground-truth state information -- useful as a baseline.

    It performs the same set of data augmentation techniques as the image-based
    policies by getting random image transform parameters, and then using those
    to adjust the ground-truth poses.

    THETA_SCALE=10 means discretizing rotations by 36, as in the CoRL submission.
    Added `one_rot_inf`, which is NOT normally true, for backwards compatibility.
    """

    def __init__(self, name, task, goal_conditioned=False, one_rot_inf=False):
        self.name = name
        self.task = task
        if self.task in ['aligning', 'palletizing', 'packing']:
            self.use_box_dimensions = True
        else:
            self.use_box_dimensions = False
        if self.task in ['sorting', 'bag-color-goal']:
            self.use_colors = True
        else:
            self.use_colors = False
        self.total_iter = 0
        self.pixel_size = 0.003125
        self.camera_config = cameras.RealSenseD415.CONFIG
        self.models_dir = os.path.join('checkpoints', self.name)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.six_dof = False
        self.goal_conditioned = goal_conditioned
        self.one_rot_inf = one_rot_inf

        # Set up model.
        self.model = None
        self.lrate = 2e-4
        self.optim = tf.keras.optimizers.Adam(learning_rate=self.lrate)
        self.metric = tf.keras.metrics.Mean(name='metric')
        self.val_metric = tf.keras.metrics.Mean(name='val_metric')
        self.THETA_SCALE = 15.0  # 24 rotations in increments of 15.
        self.BATCH_SIZE = 128
        self.USE_MDN = True

        # TODO(daniel) Hacky. I do something similar for `agents/transporter.py`.
        self.real_task = None

    def extract_x_y_theta(self, object_info, t_worldaug_world=None, preserve_theta=False, softbody=False):
        """Given either object OR action pose info, return stuff to put in GT observation.
        Note: only called from within this class and 2-step case via subclassing.

        During training, there is normally data augmentation applied, so t_worldaug_world
        is NOT None, and augmentation is applied as if the 'image' were adjusted. However,
        for actions, we preserve theta so `object_quat_xyzw` does not change, and we query
        index=2 for the z rotation. For most deformables tasks, we did not use rotations,
        hence theta=0.

        Get t_world_object which is a 4x4 homogeneous matrix, then stick the position there (why?).
        Then we do the matrix multiplication with '@' to get the homogeneous matrix representation
        of the object after augmentation is applied.
        https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication
        Then the position is extracted from the resulting homogeneous matrix. (And we ignore the
        z coordinate for that, but in fact for cloth vertices I would keep it.)

        Augmentation for cloth vertices? For now I'm following what they do for positions only
        (ignoring orientation), and returning updated (x,y,z) where the last value is the z
        (height) and not a rotation (doesn't make sense with vertrics anyway). Augmentations
        are all in SE(2), so the z coordinates should not change. Indeed that's what I see after
        extracting positions from `t_worldaug_object`.

        And yes, with vertices, they are flattened in the same way regardless of whether data
        augmentation is used, so every third value represents a vertex height. :D So we get
        [ v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, ... ] in the flattened version.

        Args:
            t_worldaug_world: Only True if there's an augmentation and we need to change the
                ground truth pose as if we applied the augmentation. Only happens during (a)
                determining mean/stds, and (b) training. During initialization and the action
                (loading) we do not use this.
            preserve_theta: Only True if calling where `object_info` is an action pose.
            softbody: Only True if `object_info` is a vertex mesh of a PyBullet soft body.

        Returns:
            object_x_y_theta, pos, quat: the first is an SE(2) pose and parameterized by
            three numbers, the xy position and a scalar rotation `theta`. In case we have
            a cloth, I'm not returning anything after that (no pos or quat) so that if we
            use the older API for cloth (which we should not be!), it should crash code.
        """
        # ------------------------------------------------------------------------------------------ #
        # Daniel: sanity checks to make sure we're handling the soft body ID correctly.
        # The last `if` of `softbody` helps to distinguish among actions, which are also length 2.
        # ------------------------------------------------------------------------------------------ #
        if (self.task in TASKS_SOFT) and (len(object_info) == 2) and softbody:
            nb_vertices, vert_pos_l = object_info
            assert nb_vertices == 100, f'We should be using 100 vertices but have: {nb_vertices}'
            assert nb_vertices == len(vert_pos_l), f'{nb_vertices}, {len(vert_pos_l)}'
            vert_pos_np = np.array(vert_pos_l)  # (100, 3)

            if t_worldaug_world is not None:
                augmented_vertices = []

                # For each vertex, apply data augmentation.
                for i in range(vert_pos_np.shape[0]):
                    vertex_position = vert_pos_np[i,:]

                    # Use identity quaternion (w=1, others 0). Othewrise, follow normal augmentation.
                    t_world_object = transformations.quaternion_matrix( (1,0,0,0) )
                    t_world_object[0:3, 3] = np.array(vertex_position)
                    t_worldaug_object = t_worldaug_world @ t_world_object
                    new_v_position = t_worldaug_object[0:3, 3]
                    augmented_vertices.append(new_v_position)

                # Stack everything.
                augmented_flattened = np.concatenate(augmented_vertices).flatten().astype(np.float32)
                return augmented_flattened
            else:
                vert_flattened = vert_pos_np.flatten().astype(np.float32)
                return vert_flattened
        # ------------------------------------------------------------------------------------------ #
        # Now proceed as usual, untouched from normal ravens code except for documentation/assertions.

        object_position = object_info[0]
        object_quat_xyzw = object_info[1]

        if t_worldaug_world is not None:
            object_quat_wxyz = (object_quat_xyzw[3], object_quat_xyzw[0], object_quat_xyzw[1], object_quat_xyzw[2])
            t_world_object = transformations.quaternion_matrix(object_quat_wxyz)
            t_world_object[0:3, 3] = np.array(object_position)

            # Daniel: pretty sure this has to be true. Upper left 3x3 is rotation, upper right 3x1 is position.
            assert t_world_object.shape == (4,4), f'shape is not 4x4: {t_world_object.shape}'
            assert t_worldaug_world.shape == (4,4), f'shape is not 4x4: {t_worldaug_world.shape}'

            # Daniel: data augmentation. Then, extract `object_position` from augmented object.
            t_worldaug_object = t_worldaug_world @ t_world_object
            object_quat_wxyz = transformations.quaternion_from_matrix(t_worldaug_object)
            if not preserve_theta:
                object_quat_xyzw = (object_quat_wxyz[1], object_quat_wxyz[2], object_quat_wxyz[3], object_quat_wxyz[0])
            object_position = t_worldaug_object[0:3, 3]

        object_xy = object_position[0:2]
        object_theta = -np.float32(utils.get_rot_from_pybullet_quaternion(object_quat_xyzw)[2]) / self.THETA_SCALE
        return np.hstack((object_xy, object_theta)).astype(np.float32), object_position, object_quat_xyzw

    def extract_box_dimensions(self, info):
        return np.array(info[2])

    def extract_color(self, info):
        # TODO(daniel) assumption is invalid, I think, if using deformable tasks. So let's change this.
        # UPDATE Oct 21: actually I changed the `info` dict for bag-color-goal so that each item (which is
        # really the `info` here) will have the rgb (not 'a' as that's always 1) info at the last part.
        # So this is fine. :D
        return np.array(info[-1])

    def info_to_gt_obs(self, info, t_worldaug_world=None, goal=None):
        """Daniel: from info dict of IDs, create the observation for GT models.

        Assumes `info` consists of just PyBullet object IDs. Creates a numpy array
        from combining the `object_x_y_theta` from all IDs, and potentially add more
        info based on if using box dimensions or colors; see `__init__()` above.

        For soft body tasks, we should have data generated so that info[cloth_id] or
        info[bag_id] contains the 3D vertex position of each point. For now we only
        test with cloth since the bag ones are honestly probably better suited with
        just the beads (but will have to check). Fortunately, all the cloth tasks as
        of late October 2020 will use ID=5 as the soft body andit's also easy to check.

        (14 Oct 2020) adding `goal` as argument for goal conditioning. Use goal['info'].
            We're going to stack the two together into one input.
        (14 Oct 2020) adding support for cloth. We don't use `pos` and `quat`, just use
            the returned object_x_y_theta.
        """
        info = self.remove_nonint_keys(info)
        if goal is not None:
            g_info = self.remove_nonint_keys(goal['info'])
        else:
            g_info = {}

        observation_vector = []
        object_keys = sorted(info.keys())
        for object_key in object_keys:
            # Daniel: adding this special case.
            if (self.task in TASKS_SOFT and len(info[object_key]) == 2):
                object_x_y_theta = self.extract_x_y_theta(info[object_key], t_worldaug_world, softbody=True)
            else:
                object_x_y_theta, pos, quat = self.extract_x_y_theta(info[object_key], t_worldaug_world)
            observation_vector.append(object_x_y_theta)
            if self.use_box_dimensions:
                observation_vector.append(self.extract_box_dimensions(info[object_key]))
            if self.use_colors:
                # For bag-color-goal we only want IDs 4 and 38.
                if self.task in ['bag-color-goal']:
                    if object_key in [4, 38]:
                        #print(f'for {object_key}, color: {self.extract_color(info[object_key])}')
                        observation_vector.append(self.extract_color(info[object_key]))
                else:
                    observation_vector.append(self.extract_color(info[object_key]))

        # Repeat for goal info using `g_info` instead of `info`, assuming g_info != {}.
        for object_key in sorted(g_info.keys()):
            # Daniel: adding this special case.
            if (self.task in TASKS_SOFT and len(g_info[object_key]) == 2):
                object_x_y_theta = self.extract_x_y_theta(g_info[object_key], t_worldaug_world, softbody=True)
            else:
                object_x_y_theta, pos, quat = self.extract_x_y_theta(g_info[object_key], t_worldaug_world)
            observation_vector.append(object_x_y_theta)
            if self.use_box_dimensions:
                observation_vector.append(self.extract_box_dimensions(g_info[object_key]))
            if self.use_colors:
                # For bag-color-goal we only want IDs 4 and 38.
                if self.task in ['bag-color-goal']:
                    if object_key in [4, 38]:
                        #print(f'for {object_key}, color: {self.extract_color(g_info[object_key])}')
                        observation_vector.append(self.extract_color(g_info[object_key]))
                else:
                    observation_vector.append(self.extract_color(g_info[object_key]))

        # Finally, make it a single 1D np.array.
        if (self.task in TASKS_SOFT):
            # Daniel: was getting some errors w/setting the type since usually we will have:
            #   [array(x1,y1,z1), array(x2,y2,z2), ...] and doing np.array() on this will return
            #   array([ [x1,y1,z1], [x2,y2,z2], ...] ) and then reshape(-1) will return
            #   array([x1,y1,z1,x2,y2,z2,...]) and it's easy to make as np.float32 type.
            # But with stuff like cloth, we have:
            #   [array(x1,y1,z1), array(c1,c2,c3,c4,...,c100), ...] and this will mean the
            # np.array().reshape(-1) results in a 2D array of arrays, so we can't do np.float32.
            # However, np.concatenate seems to fix these issues.
            observation_vector = np.concatenate(observation_vector).reshape(-1).astype(np.float32)
        else:
            observation_vector = np.array(observation_vector).reshape(-1).astype(np.float32)

        # pad with zeros
        if self.max_obs_vector_length != 0:
             observation_vector = np.pad(observation_vector,
                [0, self.max_obs_vector_length-len(observation_vector)])

        return observation_vector

    def act_to_gt_act(self, act, t_worldaug_world=None, transform_params=None):
        """Daniel: similarly, from action, create the appropriate ground truth action.

        This may involve a transformation if doing data augmentation.
        Comment from Andy/Pete: dont update theta due to suction invariance to theta
        """
        pick_se2, _, _ = self.extract_x_y_theta(act['params']['pose0'], t_worldaug_world, preserve_theta=True)
        place_se2, _, _ = self.extract_x_y_theta(act['params']['pose1'], t_worldaug_world, preserve_theta=True)
        return np.hstack((pick_se2, place_se2)).astype(np.float32)

    def set_max_obs_vector_length(self, dataset):
        """Find largest environment dimensionality.

        Daniel: likely useful for tasks such as palletizing where there's a
        variable number of objects, but I don't think it applies for my
        tasks. Make sure the dataset.path check will work, i.e., don't change
        default paths. Note:; first calls to model initialization come from
        this, can use to qwuickly debug the `info_to_gt_obs()` method as well.
        """
        if ('cable-' in dataset.path or 'insertion-goal' in dataset.path or 'cloth-' in dataset.path
                or 'bag-alone' in dataset.path or 'bag-items' in dataset.path):
            num_samples = 1
        else:
            num_samples = 2000

        self.max_obs_vector_length = 0
        max_obs_vector_length = 0
        for _ in range(num_samples):
            if self.goal_conditioned:
                _, _, info, goal = dataset.random_sample(goal_images=True)
                obs_vector_length = self.info_to_gt_obs(info, goal=goal).shape[0]
            else:
                _, _, info = dataset.random_sample()
                obs_vector_length = self.info_to_gt_obs(info).shape[0]
            if obs_vector_length > max_obs_vector_length:
                max_obs_vector_length = obs_vector_length
        self.max_obs_vector_length = max_obs_vector_length

    def init_model(self, dataset):
        """Initialize self.model, including normalization parameters."""
        self.set_max_obs_vector_length(dataset)

        # Get obs dim and action dim (3 for pick, 3 for place), initialize model.
        if self.goal_conditioned:
            _, _, info, goal = dataset.random_sample(goal_images=True)
            obs_vector = self.info_to_gt_obs(info, goal=goal)
        else:
            _, _, info = dataset.random_sample()
            obs_vector = self.info_to_gt_obs(info)

        obs_dim = obs_vector.shape[0]
        act_dim = 6
        if self.six_dof:
            act_dim = 9
        self.model = MlpModel(self.BATCH_SIZE, obs_dim, act_dim, 'relu', self.USE_MDN, dropout=0.1)

        # Sample points from the data to get reasonable mean / std values.
        sampled_gt_obs = []
        num_samples = 1000
        for _ in range(num_samples):
            t_worldaug_world, _ = self.get_augmentation_transform()
            if self.goal_conditioned:
                _, _, info, goal = dataset.random_sample(goal_images=True)
                sampled_gt_obs.append(self.info_to_gt_obs(info, t_worldaug_world, goal=goal))
            else:
                _, _, info = dataset.random_sample()
                sampled_gt_obs.append(self.info_to_gt_obs(info, t_worldaug_world))
        sampled_gt_obs = np.array(sampled_gt_obs)
        obs_train_parameters = dict()
        obs_train_parameters['mean'] = sampled_gt_obs.mean(axis=(0)).astype(np.float32)
        obs_train_parameters['std'] = sampled_gt_obs.std(axis=(0)).astype(np.float32)
        self.model.set_normalization_parameters(obs_train_parameters)

        self.obs_train_mean = obs_train_parameters['mean']
        self.obs_train_std  = obs_train_parameters['std']
        print('Done initializing self.model for ground truth agent.')

    def get_augmentation_transform(self):
        heightmap = np.zeros((320,160))
        theta, trans, pivot = utils.get_random_image_transform_params(heightmap.shape)
        transform_params = theta, trans, pivot
        t_world_center, t_world_centeraug = utils.get_se3_from_image_transform(*transform_params,
            heightmap, self.bounds, self.pixel_size)
        t_worldaug_world = t_world_centeraug @ np.linalg.inv(t_world_center)
        return t_worldaug_world, transform_params

    def get_data_batch(self, dataset):
        """Pre-process info and obs-act, and make batch.
        Daniel: adding goal-conditioning. To make it easier, just stack observations.
        """
        batch_obs = []
        batch_act = []
        for _ in range(self.BATCH_SIZE):
            t_worldaug_world, transform_params = self.get_augmentation_transform()
            if self.goal_conditioned:
                obs, act, info, goal = dataset.random_sample(goal_images=True)
                gt_obs = self.info_to_gt_obs(info, t_worldaug_world, goal=goal)
            else:
                obs, act, info = dataset.random_sample()
                gt_obs = self.info_to_gt_obs(info, t_worldaug_world)
            batch_obs.append(gt_obs)
            batch_act.append(self.act_to_gt_act(act, t_worldaug_world, transform_params)) # this samples pick points from surface
            # on insertion task only, this can be used to imagine as if the picks were deterministic
            # batch_act.append(self.info_to_gt_obs(info))
        batch_obs = np.array(batch_obs)
        batch_act = np.array(batch_act)
        return batch_obs, batch_act, obs, act, info

    def train(self, dataset, num_iter, writer, validation_dataset=None):
        """Train on dataset for a specific number of iterations.

        Daniel: not testing with validation, argument copied over from ravens. Naively,
        we can train with MSE, but better to use a mixture model (MDN) since the output
        should be multi-modal; could be several pick points, and several placing points
        wrt those pick points.

        Also, notice how one iteration involves taking a batch of data, whereas for
        Transporters, we only used one image per 'iteration'. This means that each gt-state
        model consumes 128x more data points during training.

        NOTE: there is a special case for num_iter=0 which I use for the "zero iteration" baseline.
        Without this, it is impossible to save the model because it hasn't been built yet. However,
        even with this I get the warning:

        WARNING:tensorflow:Skipping full serialization of Keras layer <tensorflow.python.keras.layers.core.Dense object at 0x7fc88c170890>, because it is not built.

        but the loading seems to be OK, so I'm not sure what could be the issue, and given that
        it's for the 0 iteration case, it's unlikely to be worth fully investigating for now.
        """
        if self.model is None:
            self.init_model(dataset)

        if self.USE_MDN:
            loss_criterion = mdn_utils.mdn_loss
        else:
            loss_criterion = tf.keras.losses.MeanSquaredError()

        @tf.function
        def train_step(model, batch_obs, batch_act, loss_criterion):
            with tf.GradientTape() as tape:
                prediction = model(batch_obs)
                loss = loss_criterion(batch_act, prediction)
                grad = tape.gradient(loss, model.trainable_variables)
                self.optim.apply_gradients(zip(grad, model.trainable_variables))
            return loss

        print_rate = 50
        VALIDATION_RATE = 1000

        # Need this case due to some quirks with saving this type of model. No gradient tupdates.
        if num_iter == 0:
            batch_obs, batch_act, obs, act, info = self.get_data_batch(dataset)
            print('Doing a single forward pass to enable us to save a snapshot for the num_iter == 0 case.')
            prediction = self.model(batch_obs)

        for i in range(num_iter):
            start = time.time()
            batch_obs, batch_act, obs, act, info = self.get_data_batch(dataset)

            # Forward through model, compute training loss, update weights.
            self.metric.reset_states()
            loss = train_step(self.model, batch_obs, batch_act, loss_criterion)
            self.metric(loss)
            with writer.as_default():
                tf.summary.scalar('gt_state_loss', self.metric.result(), step=self.total_iter+i)
            if i % print_rate == 0:
                loss = np.float32(loss)
                print(f'Train Iter: {self.total_iter + i} Loss: {loss:.4f} Iter time:', time.time() - start)

            # Compute valid loss only if we have a validation dataset.
            if ((self.total_iter + i) % VALIDATION_RATE == 0) and (validation_dataset is not None):
                print("Validating!")
                tf.keras.backend.set_learning_phase(0)
                self.val_metric.reset_states()
                batch_obs, batch_act, _, _, _ = self.get_data_batch(validation_dataset)
                prediction = self.model(batch_obs)
                loss = loss_criterion(batch_act, prediction)
                self.val_metric(loss)
                with writer.as_default():
                    tf.summary.scalar('validation_gt_state_loss', self.val_metric.result(),
                        step=self.total_iter+i)
                tf.keras.backend.set_learning_phase(1)

        self.total_iter += num_iter
        self.save()

    def plot_act_mdn(self, y, mdn_predictions):
        """
        Args:
            y: true "y", shape (batch_size, d_out)
            mdn_predictions: tuple of:
                pi: (batch_size, num_gaussians)
                mu: (batch_size, num_gaussians * d_out)
                var: (batch_size, num_gaussians)
        """
        pi, mu, var = mdn_predictions

        self.ax.cla()
        self.ax.scatter(y[:,0], y[:,1])
        mu = tf.reshape(mu, (-1, y.shape[-1]))
        pi = tf.reshape(pi, (-1,))
        pi = tf.clip_by_value(pi, 0.01, 1.0)

        rgba_colors = np.zeros((len(pi),4))
        # for red the first column needs to be one
        rgba_colors[:,0] = 1.0
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = pi
        self.ax.scatter(mu[:,0], mu[:,1], color=rgba_colors)

        plt.draw()
        plt.pause(0.001)

    def act(self, obs, info, goal=None):
        """Run inference and return best action."""
        act = {'camera_config': self.camera_config, 'primitive': None}

        # Get observations and run predictions (second part just for visualization).
        if self.goal_conditioned:
            gt_obs = self.info_to_gt_obs(info, goal=goal)
            gt_act_center = self.info_to_gt_obs(info, goal=goal)
        else:
            gt_obs = self.info_to_gt_obs(info)
            gt_act_center = self.info_to_gt_obs(info)

        prediction = self.model(gt_obs[None, ...])

        if self.USE_MDN:
            mdn_prediction = prediction
            pi, mu, var = mdn_prediction
            #prediction = mdn_utils.pick_max_mean(pi, mu, var)
            prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            prediction = prediction[:, 0, :]

        prediction = prediction[0] # unbatch

        # Just go exactly to objects, predicted. Daniel: adding 1 rotation inference case.
        p0_position = np.hstack((prediction[0:2], 0.02))
        p0_pred_rot = 0.0 if self.one_rot_inf else -prediction[2]*self.THETA_SCALE  # idx 2
        p0_rotation = utils.get_pybullet_quaternion_from_rot((0, 0, p0_pred_rot))
        p1_position = np.hstack((prediction[3:5], 0.02))
        p1_pred_rot = 0.0 if self.one_rot_inf else -prediction[5]*self.THETA_SCALE  # idx 5
        p1_rotation = utils.get_pybullet_quaternion_from_rot((0, 0, p1_pred_rot))

        # Select task-specific motion primitive.
        act['primitive'] = 'pick_place'
        if self.task == 'sweeping':
            act['primitive'] = 'sweep'
        elif self.task == 'pushing':
            act['primitive'] = 'push'
        params = {'pose0': (p0_position, p0_rotation),
                  'pose1': (p1_position, p1_rotation)}
        act['params'] = params

        # Daniel: like transporters, determine the task stage if applicable. (AND if loading only)
        if self.task in ['bag-items-easy', 'bag-items-hard', 'bag-color-goal']:
            self._determine_task_stage(p0_position, p1_position)

        return act

    #-------------------------------------------------------------------------
    # Helper Functions. Since we're subclassing self.model, rather than directly
    # using self.model = tf.keras.Model(), we need to change saving/loading a bit.
    # Use tf save format (not h5) and use `keras.models.load_model()``.
    # NOTE: requires TensorFlow 2.2 or later.
    #-------------------------------------------------------------------------

    def load(self, num_iter):
        """Load parameters.

        For this, set compile=False because we're not retraining the model.
        Also, this requires the max_obs_vector_length restored, as that isn't
        in the `__init__()`.
        """
        model_fname = 'gt-state-ckpt-%d' % num_iter
        model_fname = os.path.join(self.models_dir, model_fname)
        self.model = keras.models.load_model(model_fname, compile=False)
        self.total_iter = num_iter

        # Load other data we need for proper usage of the model.
        data_fname = os.path.join(self.models_dir, 'misc_data.pkl')
        with open(data_fname, 'rb') as fh:
            data = pickle.load(fh)
        self.max_obs_vector_length = data['max_obs_vector_length']

        # Note: we cannot call self.model methods directly, other than __call__.
        # Actually, I'm not even sure if this works, but we don't normalize
        # with the current model. TODO(daniel) does this work?
        self.model.obs_train_mean = data['obs_train_mean']
        self.model.obs_train_std = data['obs_train_std']

    def save(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        model_fname = 'gt-state-ckpt-%d' % self.total_iter
        model_fname = os.path.join(self.models_dir, model_fname)
        self.model.save(model_fname, save_format='tf')
        print(f'Saving model: {model_fname}')

        # Save other data we need for proper usage of the model.
        data_fname = os.path.join(self.models_dir, 'misc_data.pkl')
        data = dict(max_obs_vector_length=self.max_obs_vector_length,
                    obs_train_mean=self.obs_train_mean,
                    obs_train_std=self.obs_train_std)
        with open(data_fname, 'wb') as fh:
            pickle.dump(data, fh)

    def remove_nonint_keys(self, info):
        return {k:info[k] for k in info if isinstance(k, int)}

    def _determine_task_stage(self, p0_position, p1_position):
        """Determines task stage for the bag-items tasks, for gt_state and gt_state_2_step.

        See agents/transporter.py for details. The ONLY difference here is that we have
        the positions and need to do a position to pixel conversion, which is trivial with our
        utility file. Otherwise, we follow the same procedure as in the transporter class, with
        the same set of drawbacks / caveats.
        """
        p0_pixel = utils.position_to_pixel(p0_position, bounds=self.bounds, pixel_size=self.pixel_size)
        p1_pixel = utils.position_to_pixel(p1_position, bounds=self.bounds, pixel_size=self.pixel_size)

        # Daniel: hack to get things in the bounds.
        p0_x = min( max(p0_pixel[0], 0), 319)
        p0_y = min( max(p0_pixel[1], 0), 159)
        p0_pixel = (int(p0_x), int(p0_y))

        real_task = self.real_task  # assume we assigned this.
        colormap, heightmap, object_mask = real_task.get_object_masks(real_task.env)

        if False:
            nb = len([x for x in os.listdir('.') if '.png' in x])
            mask = np.array(object_mask / np.max(object_mask) * 255).astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # debugging
            p0 = (p0_pixel[1], p0_pixel[0])
            p1 = (p1_pixel[1], p1_pixel[0])
            cv2.circle(mask, p0, radius=3, color=(255,0,255), thickness=-1)
            cv2.circle(mask, p1, radius=3, color=(255,255,0), thickness=-1)
            cv2.imwrite(f'mask_{nb}.png', mask)

        # Copied from ravens/agents/transporter.py.
        if self.task in ['bag-items-easy', 'bag-items-hard']:
            if object_mask[p0_pixel] in [38, 39]:
                real_task.task_stage = 2
            elif real_task.task_stage == 2:
                real_task.task_stage = 3
        elif self.task in ['bag-color-goal']:
            if object_mask[p0_pixel] == real_task.single_block_ID:
                real_task.task_stage = 2
            else:
                real_task.task_stage = 1
        else:
            raise NotImplementedError(self.task)
