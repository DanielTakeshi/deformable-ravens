#!/usr/bin/env python
import os
import cv2
import time
import pickle
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import tensorflow as tf; tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from ravens import cameras
from ravens import utils
from ravens.agents import GtStateAgent
from ravens.models import MlpModel
from ravens.models import mdn_utils


class GtState2StepAgent(GtStateAgent):
    """Agent which uses ground-truth state information -- useful as a baseline.

    It calls the superclass (the one-step GT agent) for getting the data batch, and
    the subsequent data augmentation. See docs in GtStateAgent.
    """

    def __init__(self, name, task, goal_conditioned=False, one_rot_inf=False):
        super(GtState2StepAgent, self).__init__(name, task, goal_conditioned, one_rot_inf)

        # Set up model.
        self.pick_model = None
        self.place_model = None
        self.pick_lrate = 2e-4
        self.place_lrate = 2e-4
        self.pick_optim = tf.keras.optimizers.Adam(learning_rate=self.pick_lrate)
        self.place_optim = tf.keras.optimizers.Adam(learning_rate=self.place_lrate)
        self.metric = tf.keras.metrics.Mean(name='metric')
        self.val_metric = tf.keras.metrics.Mean(name='val_metric')

    def init_model(self, dataset):
        """Initialize models, including normalization parameters."""
        self.set_max_obs_vector_length(dataset)

        if self.goal_conditioned:
            _, _, info, goal = dataset.random_sample(goal_images=True)
            obs_vector = self.info_to_gt_obs(info, goal=goal)
        else:
            _, _, info = dataset.random_sample()
            obs_vector = self.info_to_gt_obs(info)

        # Setup pick model, which only has act_dim=3 unlike act_dim=6 for gt_state.
        obs_dim = obs_vector.shape[0]
        act_dim = 3
        self.pick_model = MlpModel(self.BATCH_SIZE, obs_dim, act_dim, 'relu', self.USE_MDN, dropout=0.1)

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
        obs_train_parameters['std']  = sampled_gt_obs.std(axis=(0)).astype(np.float32)
        self.pick_model.set_normalization_parameters(obs_train_parameters)
        self.pick_obs_train_mean = obs_train_parameters['mean']
        self.pick_obs_train_std  = obs_train_parameters['std']

        # Setup pick-conditioned place model, which adds `act_dim` to `obs_dim`.
        obs_dim = obs_vector.shape[0] + act_dim
        act_dim = 3
        self.place_model = MlpModel(self.BATCH_SIZE, obs_dim, act_dim, 'relu', self.USE_MDN, dropout=0.1)

        # Sample points from the data to get reasonable mean / std values.
        sampled_gt_obs = []
        num_samples = 1000
        for _ in range(num_samples):
            t_worldaug_world, _ = self.get_augmentation_transform()
            if self.goal_conditioned:
                _, act, info, goal = dataset.random_sample(goal_images=True)
                obs = self.info_to_gt_obs(info, t_worldaug_world, goal=goal)
            else:
                _, act, info = dataset.random_sample()
                obs = self.info_to_gt_obs(info, t_worldaug_world)
            obs = np.hstack((obs, self.act_to_gt_act(act, t_worldaug_world)[:3]))  # Daniel: key difference?
            sampled_gt_obs.append(obs)
        sampled_gt_obs = np.array(sampled_gt_obs)
        obs_train_parameters = dict()
        obs_train_parameters['mean'] = sampled_gt_obs.mean(axis=(0)).astype(np.float32)
        obs_train_parameters['std']  = sampled_gt_obs.std(axis=(0)).astype(np.float32)
        self.place_model.set_normalization_parameters(obs_train_parameters)
        self.place_obs_train_mean = obs_train_parameters['mean']
        self.place_obs_train_std  = obs_train_parameters['std']

        print('Done initializing self.model for ground truth agent.')

    def train(self, dataset, num_iter, writer, validation_dataset=None):
        """Train on dataset for a specific number of iterations.

        As with the gt_state, need a special case to handle the num_iter=0 case.
        """
        if self.pick_model is None:
            self.init_model(dataset)

        if self.USE_MDN:
            loss_criterion = mdn_utils.mdn_loss
        else:
            loss_criterion = tf.keras.losses.MeanSquaredError()

        @tf.function
        def train_step(pick_model, place_model, batch_obs, batch_act, loss_criterion):
            with tf.GradientTape() as tape:
                prediction = pick_model(batch_obs)
                loss0 = loss_criterion(batch_act[:,0:3], prediction)
                grad = tape.gradient(loss0, pick_model.trainable_variables)
                self.pick_optim.apply_gradients(zip(grad, pick_model.trainable_variables))
            with tf.GradientTape() as tape:
                #batch_obs = tf.concat((batch_obs, batch_act[:,0:3] + tf.random.normal(shape=batch_act[:,0:3].shape, stddev=0.001)), axis=1)
                batch_obs = tf.concat((batch_obs, batch_act[:,0:3]), axis=1)
                prediction = place_model(batch_obs)
                loss1 = loss_criterion(batch_act[:,3:], prediction)
                grad = tape.gradient(loss1, place_model.trainable_variables)
                self.place_optim.apply_gradients(zip(grad, place_model.trainable_variables))
            return loss0 + loss1

        # Need this case due to some quirks with saving this type of model. No gradient tupdates.
        if num_iter == 0:
            batch_obs, batch_act, obs, act, info = self.get_data_batch(dataset)
            print('Doing a single forward pass to enable us to save a snapshot for the num_iter == 0 case.')
            prediction = self.pick_model(batch_obs)
            batch_obs = tf.concat((batch_obs, batch_act[:,0:3]), axis=1)
            prediction = self.place_model(batch_obs)

        print_rate = 50
        for i in range(num_iter):
            start = time.time()
            batch_obs, batch_act, obs, act, info = self.get_data_batch(dataset)

            # Forward through model, compute training loss, update weights.
            self.metric.reset_states()
            loss = train_step(self.pick_model, self.place_model, batch_obs, batch_act, loss_criterion)
            self.metric(loss)
            with writer.as_default():
                tf.summary.scalar('gt_state_loss', self.metric.result(), step=self.total_iter+i)
            if i % print_rate == 0:
                loss = np.float32(loss)
                print(f'Train Iter: {self.total_iter + i} Loss: {loss:.4f} Iter time:', time.time() - start)

        self.total_iter += num_iter
        self.save()

    def act(self, obs, info, goal=None):
        """Run inference and return best action."""
        act = {'camera_config': self.camera_config, 'primitive': None}

        # Get observations and run pick prediction
        if self.goal_conditioned:
            gt_obs = self.info_to_gt_obs(info, goal=goal)
        else:
            gt_obs = self.info_to_gt_obs(info)
        pick_prediction = self.pick_model(gt_obs[None, ...])
        if self.USE_MDN:
            pi, mu, var = pick_prediction
            #prediction = mdn_utils.pick_max_mean(pi, mu, var)
            pick_prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            pick_prediction = pick_prediction[:, 0, :]
        pick_prediction = pick_prediction[0] # unbatch

        # Get observations and run place prediction
        obs_with_pick = np.hstack((gt_obs, pick_prediction))

        # since the pick at train time is always 0.0,
        # the predictions are unstable if not exactly 0
        obs_with_pick[-1] = 0.0

        place_prediction = self.place_model(obs_with_pick[None, ...])
        if self.USE_MDN:
            pi, mu, var = place_prediction
            #prediction = mdn_utils.pick_max_mean(pi, mu, var)
            place_prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            place_prediction = place_prediction[:, 0, :]
        place_prediction = place_prediction[0]

        prediction = np.hstack((pick_prediction, place_prediction))

        # Daniel: like with gt_state, guessing this is just for insertion.
        # just go exactly to objects, from observations
        # p0_position = np.hstack((gt_obs[3:5], 0.02))
        # p0_rotation = utils.get_pybullet_quaternion_from_rot((0, 0, -gt_obs[5]*self.THETA_SCALE))
        # p1_position = np.hstack((gt_obs[0:2], 0.02))
        # p1_rotation = utils.get_pybullet_quaternion_from_rot((0, 0, -gt_obs[2]*self.THETA_SCALE))

        # Just go exactly to objects, predicted. Daniel: adding 1 rotation  inference case.
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
        if self.task in ['bag-items-easy', 'bag-items-hard']:
            self._determine_task_stage(p0_position, p1_position)

        return act

    #-------------------------------------------------------------------------
    # Helper Functions. See gt_state, we need to do something similar here.
    #-------------------------------------------------------------------------

    def load(self, num_iter):
        """Load in a similar fashion as the 1-step GT agent."""
        pick_fname  = 'gt-state-2-step-pick-ckpt-%d' % num_iter
        place_fname = 'gt-state-2-step-place-ckpt-%d' % num_iter
        pick_fname  = os.path.join(self.models_dir, pick_fname)
        place_fname = os.path.join(self.models_dir, place_fname)
        self.pick_model  = keras.models.load_model(pick_fname, compile=False)
        self.place_model = keras.models.load_model(place_fname, compile=False)
        self.total_iter = num_iter

        # Load other data we need for proper usage of the model.
        data_fname = os.path.join(self.models_dir, 'misc_data.pkl')
        with open(data_fname, 'rb') as fh:
            data = pickle.load(fh)
        self.max_obs_vector_length = data['max_obs_vector_length']

        # Note: we cannot call self.model methods directly, other than __call__.
        # Actually, I'm not even sure if this works, but we don't normalize
        # with the current model. TODO(daniel) does this work?
        self.pick_model.obs_train_mean  = data['pick_obs_train_mean']
        self.place_model.obs_train_mean = data['place_obs_train_mean']
        self.pick_model.obs_train_std   = data['pick_obs_train_std']
        self.place_model.obs_train_std  = data['place_obs_train_std']

    def save(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        pick_fname  = 'gt-state-2-step-pick-ckpt-%d' % self.total_iter
        place_fname = 'gt-state-2-step-place-ckpt-%d' % self.total_iter
        pick_fname  = os.path.join(self.models_dir, pick_fname)
        place_fname = os.path.join(self.models_dir, place_fname)
        self.pick_model.save(pick_fname, save_format='tf')
        self.place_model.save(place_fname, save_format='tf')
        print(f'Saving model: {pick_fname}')
        print(f'Saving model: {place_fname}')

        # Save other data we need for proper usage of the model.
        data_fname = os.path.join(self.models_dir, 'misc_data.pkl')
        data = dict(max_obs_vector_length=self.max_obs_vector_length,
                    pick_obs_train_mean  =self.pick_obs_train_mean,
                    place_obs_train_mean =self.place_obs_train_mean,
                    pick_obs_train_std   =self.pick_obs_train_std,
                    place_obs_train_std  =self.place_obs_train_std)
        with open(data_fname, 'wb') as fh:
            pickle.dump(data, fh)
