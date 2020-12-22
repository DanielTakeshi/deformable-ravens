#!/usr/bin/env python

import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from ravens.models import ResNet43_8s, ConvMLP
from ravens import utils



class Regression:

    def __init__(self, input_shape, preprocess):
        self.preprocess = preprocess
        

        RESNET = False
        if RESNET:
            output_dim = 6
            in0, out0 = ResNet43_8s(input_shape, output_dim, prefix='s0_')

            out0 = tf.nn.avg_pool(out0, ksize=(1,4,4,1), strides=(1,4,4,1), padding="SAME", data_format="NHWC")
            
            out0 = tf.keras.layers.Flatten()(out0)
            out0 = tf.keras.layers.Dense(128,
                                         kernel_initializer="normal",
                                         bias_initializer="normal",
                                         activation='relu')(out0)
            out0 = tf.keras.layers.Dense(3,
                                         kernel_initializer="normal",
                                         bias_initializer="normal")(out0)

            self.model = tf.keras.Model(inputs=[in0], outputs=[out0])

        else:
            self.model = ConvMLP(d_action=3)

        self.optim = tf.keras.optimizers.Adam()
        self.metric = tf.keras.metrics.Mean(name='regression_loss')
        self.loss_criterion = tf.keras.losses.MeanSquaredError()

    def set_batch_size(self, batch_size):
        self.model.set_batch_size(batch_size)

    def forward(self, in_img):
        """Forward pass.

        in_img: [B, H, W, C]
        """
        input_data = self.preprocess(in_img)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        output = self.model(in_tensor)
        return output

    def train_pick(self, in_img, p, theta, train_step):
        """
          Regress pixel p.
          Args:
            in_img: shape (B, H, W, C)
            p: pixel (y, x), shape (B, 2)
            theta: angle, shape (B, 1)
        """
        self.metric.reset_states()
        yxtheta = np.array([p[:,0], p[:,1], theta]).T
        input_data = self.preprocess(in_img)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        loss = train_step(self.model, self.optim, in_tensor, yxtheta, self.loss_criterion)
        self.metric(loss)
        return np.float32(loss)

    def train_place_conditioned_on_pick(self, in_img, p, q, theta):
        """
          Regress pixel p to pixel q.
          Args:
            input:
            p: pixel (y, x)
            q: pixel (y, x)
        """
        self.metric.reset_states()
        with tf.GradientTape() as tape:
            output = self.forward(in_img)
            delta_pixel = np.array(q) - np.array(p)
            yxtheta = np.array([delta_pixel[0], delta_pixel[1], theta])
            loss = self.loss_criterion(yxtheta, output)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grad, self.model.trainable_variables))

        self.metric(loss)
        return np.float32(loss)


    def save(self, fname):
        pass

    def load(self, fname):
        pass