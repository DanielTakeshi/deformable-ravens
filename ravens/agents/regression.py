#!/usr/bin/env python

import os
import time

import cv2
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import tensorflow as tf; tf.compat.v1.enable_eager_execution()

from ravens.models import Attention, Regression
from ravens import cameras
from ravens import utils


class RegressionAgent:

    def __init__(self, name, task):
        self.name = name
        self.task = task
        self.total_iter = 0
        self.pixel_size = 0.003125
        self.input_shape = (320, 160, 6)
        self.camera_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.total_iter = 0

        # A place to save pre-trained models.
        self.models_dir = os.path.join('checkpoints', self.name)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Set up model.
        self.optim = tf.keras.optimizers.Adam(lr=1e-2)
        self.metric = tf.keras.metrics.Mean(name='metric')

        self.batch_size = 4

    def show_images(self, colormap, heightmap):
        import matplotlib.pyplot as plt
        plt.imshow(colormap)
        plt.show()
        plt.imshow(heightmap)
        plt.show()

    def train(self, dataset, num_iter, writer):
        """Train on dataset for a specific number of iterations."""

        @tf.function
        def pick_train_step(model, optim, in_tensor, yxtheta, loss_criterion):
            with tf.GradientTape() as tape:
                output = model(in_tensor)
                loss = loss_criterion(yxtheta, output)
            grad = tape.gradient(loss, model.trainable_variables)
            optim.apply_gradients(
                zip(grad, model.trainable_variables))
            return loss

        @tf.function
        def place_train_step(model, optim, in_tensor, yxtheta, loss_criterion):
            with tf.GradientTape() as tape:
                output = model(in_tensor)
                loss = loss_criterion(yxtheta, output)
            grad = tape.gradient(loss, model.trainable_variables)
            optim.apply_gradients(
                zip(grad, model.trainable_variables))
            return loss

        for i in range(num_iter):
            start = time.time()


            input_images, p0s, p0_thetas = [], [], []
            p1s, p1_thetas = [], []
            for _ in range(self.batch_size):
                obs, act, info = dataset.random_sample()

                # Get heightmap from RGB-D images.
                configs = act['camera_config']
                colormap, heightmap = self.get_heightmap(obs, configs)
                #self.show_images(colormap, heightmap)

                # Get training labels from data sample.

                # (spatially distributed on object) get actions from oracle distribution
                #pose0, pose1 = act['params']['pose0'], act['params']['pose1']

                # (identical object location) get actions from object poses
                l_object = info[4]
                pose0 = l_object[0], l_object[1]
                l_target = info[5]
                pose1 = l_target[0], l_target[1]

                p0_position, p0_rotation = pose0[0], pose0[1]
                p0 = utils.position_to_pixel(p0_position, self.bounds, self.pixel_size)
                p0_theta = -np.float32(p.getEulerFromQuaternion(p0_rotation)[2])
                p1_position, p1_rotation = pose1[0], pose1[1]
                p1 = utils.position_to_pixel(p1_position, self.bounds, self.pixel_size)
                p1_theta = -np.float32(p.getEulerFromQuaternion(p1_rotation)[2])

                # to make it relative
                # p1_theta = p1_theta - p0_theta
                # p0_theta = 0

                p1_xytheta = np.array([p1_position[0], p1_position[1], p1_theta])

                # Concatenate color with depth images.
                # input_image = np.concatenate((colormap,
                #                               heightmap[..., None],
                #                               heightmap[..., None],
                #                               heightmap[..., None]), axis=2)

                input_image = colormap

                input_images.append(input_image)
                p0s.append(p0)
                p0_thetas.append(p0_theta)
                p1s.append(p1)
                p1_thetas.append(p1_theta)

            input_image = np.array(input_images)
            p0 = np.array(p0s)
            p0_theta = np.array(p0_thetas)
            p1 = np.array(p1s)
            p1_theta = np.array(p1_thetas)

            # Compute train loss - regression place
            loss0 = self.pick_regression_model.train_pick(input_image, p0, p0_theta, pick_train_step)
            with writer.as_default():
                tf.summary.scalar('pick_loss', self.pick_regression_model.metric.result(),
                    step=self.total_iter+i)

            # Compute train loss - regression place
            loss1 = self.place_regression_model.train_pick(input_image, p1, p1_theta, place_train_step)
            with writer.as_default():
                tf.summary.scalar('place_loss', self.place_regression_model.metric.result(),
                    step=self.total_iter+i)

            #loss1 = 0.0
            print(f'Train Iter: {self.total_iter + i} Loss: {loss0:.4f} {loss1:.4f} Iter time:', time.time() - start)

        self.total_iter += num_iter
        self.save()

    def act(self, obs, info):
        """Run inference and return best action given visual observations."""
        self.pick_regression_model.set_batch_size(1)
        self.place_regression_model.set_batch_size(1)
        act = {'camera_config': self.camera_config, 'primitive': None}
        if not obs:
            return act

        # Get heightmap from RGB-D images.
        colormap, heightmap = self.get_heightmap(obs, self.camera_config)

        # Concatenate color with depth images.
        # input_image = np.concatenate((colormap,
        #                               heightmap[..., None],
        #                               heightmap[..., None],
        #                               heightmap[..., None]), axis=2)

        input_image = colormap[None, ...]

        # Regression pick model
        p0_yxtheta = self.pick_regression_model.forward(input_image)[0] # unbatch
        p0_pixel = [int(p0_yxtheta[0]), int(p0_yxtheta[1])]
        p0_theta = p0_yxtheta[2]

        # Regression place model
        p1_yxtheta = self.place_regression_model.forward(input_image)[0] # unbatch
        p1_pixel = [int(p1_yxtheta[0]), int(p1_yxtheta[1])]
        p1_theta = p1_yxtheta[2]

        # make sure safe:
        if p1_pixel[0] < 0:
            p1_pixel[0] = 0
        if p1_pixel[0] > 319:
            p1_pixel[0] = 319

        if p1_pixel[1] < 0:
            p1_pixel[1] = 0
        if p1_pixel[1] > 159:
            p1_pixel[1] = 159

        # Pixels to end effector poses.
        p0_position = utils.pixel_to_position(p0_pixel, heightmap, self.bounds, self.pixel_size)
        p1_position = utils.pixel_to_position(p1_pixel, heightmap, self.bounds, self.pixel_size)

        p0_rotation = p.getQuaternionFromEuler((0, 0, -p0_theta))
        p1_rotation = p.getQuaternionFromEuler((0, 0, -p1_theta))

        act['primitive'] = 'pick_place'
        if self.task == 'sweeping':
            act['primitive'] = 'sweep'
        elif self.task == 'pushing':
            act['primitive'] = 'push'
        params = {'pose0': (p0_position, p0_rotation),
                  'pose1': (p1_position, p1_rotation)}
        act['params'] = params
        self.pick_regression_model.set_batch_size(self.batch_size)
        self.place_regression_model.set_batch_size(self.batch_size)
        return act


    #-------------------------------------------------------------------------
    # Helper Functions
    #-------------------------------------------------------------------------

    def preprocess(self, image):
        """Pre-process images (subtract mean, divide by std).
        image shape: [B, H, W, C]
        """
        color_mean = 0.18877631
        depth_mean = 0.00509261
        color_std = 0.07276466
        depth_std = 0.00903967
        image[:, :, :, :3] = (image[:, :, :, :3] / 255 - color_mean) / color_std
        #image[:, :, :, 3:] = (image[:, :, :, 3:] - depth_mean) / depth_std
        return image

    def get_heightmap(self, obs, configs):
        """Reconstruct orthographic heightmaps with segmentation masks."""
        heightmaps, colormaps = utils.reconstruct_heightmaps(
            obs['color'], obs['depth'], configs, self.bounds, self.pixel_size)
        colormaps = np.float32(colormaps)
        heightmaps = np.float32(heightmaps)

        # Fuse maps from different views.
        valid = np.sum(colormaps, axis=3) > 0
        repeat = np.sum(valid, axis=0)
        repeat[repeat == 0] = 1
        colormap = np.sum(colormaps, axis=0) / repeat[..., None]
        colormap = np.uint8(np.round(colormap))
        heightmap = np.max(heightmaps, axis=0)
        return colormap, heightmap

    def load(self, num_iter):
        pass

    def save(self):
        pass


class PickThenPlaceRegressionAgent(RegressionAgent):

    def __init__(self, name, task):
        super().__init__(name, task)

        self.pick_regression_model = Regression(input_shape=self.input_shape,
                                           preprocess=self.preprocess)
        self.pick_regression_model.set_batch_size(self.batch_size)

        self.place_regression_model = Regression(input_shape=self.input_shape,
                                           preprocess=self.preprocess)
        self.place_regression_model.set_batch_size(self.batch_size)