#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from ravens.models import ResNet43_8s
from ravens import utils


class Attention:
    """Daniel: attention model implemented as an hourglass FCN.

    Used for the picking network, and for placing if doing the 'no-transport'
    ablation. By default our TransporterAgent class (using this for picking)
    has num_rotations=1, leaving rotations to the Transport component. Input
    shape is (320,160,6) with 3 height maps, and then we change it so the H
    and W are both 320 (to support rotations).

    In the normal Transporter model, this component only uses one rotation,
    so the label is just sized at (320,160,1).
    """

    def __init__(self, input_shape, num_rotations, preprocess):
        self.num_rotations = num_rotations
        self.preprocess = preprocess

        max_dim = np.max(input_shape[:2])

        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(input_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        input_shape = np.array(input_shape)
        input_shape += np.sum(self.padding, axis=1)
        input_shape = tuple(input_shape)

        # Initialize fully convolutional Residual Network with 43 layers and
        # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
        d_in, d_out = ResNet43_8s(input_shape, 1)
        self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
        self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.metric = tf.keras.metrics.Mean(name='attention_loss')

    def forward(self, in_img, apply_softmax=True):
        """Forward pass.

        in_img.shape: (320, 160, 6)
        input_data.shape: (320, 320, 6), then (None, 320, 320, 6)
        """
        input_data = np.pad(in_img, self.padding, mode='constant')
        input_data = self.preprocess(input_data)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)
        in_tens = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Rotate input
        pivot = np.array(input_data.shape[1:3]) / 2
        rvecs = self.get_se2(self.num_rotations, pivot)
        in_tens = tf.repeat(in_tens, repeats=self.num_rotations, axis=0)
        # https://www.tensorflow.org/addons/api_docs/python/tfa/image/transform
        in_tens = tfa.image.transform(in_tens, rvecs, interpolation='NEAREST')

        # Forward pass
        in_tens = tf.split(in_tens, self.num_rotations)
        logits = ()
        for x in in_tens:
            logits += (self.model(x),)
        logits = tf.concat(logits, axis=0)

        # Rotate back output
        rvecs = self.get_se2(self.num_rotations, pivot, reverse=True)
        logits = tfa.image.transform(logits, rvecs, interpolation='NEAREST')
        c0 = self.padding[:2, 0]
        c1 = c0 + in_img.shape[:2]
        logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

        logits = tf.transpose(logits, [3, 1, 2, 0])
        output = tf.reshape(logits, (1, np.prod(logits.shape)))
        if apply_softmax:
            output = np.float32(output).reshape(logits.shape[1:])
        return output

    def train(self, in_img, p, theta):
        self.metric.reset_states()
        with tf.GradientTape() as tape:
            output = self.forward(in_img, apply_softmax=False)

            # Compute label
            theta_i = theta / (2 * np.pi / self.num_rotations)
            theta_i = np.int32(np.round(theta_i)) % self.num_rotations
            label_size = in_img.shape[:2] + (self.num_rotations,)
            label = np.zeros(label_size)
            label[p[0], p[1], theta_i] = 1
            label = label.reshape(1, np.prod(label.shape))
            label = tf.convert_to_tensor(label, dtype=tf.float32)

            # Compute loss
            loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
            loss = tf.reduce_mean(loss)

        # Backpropagate
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(
            zip(grad, self.model.trainable_variables))

        self.metric(loss)
        return np.float32(loss)

    def load(self, path):
        self.model.load_weights(path)

    def save(self, filename):
        self.model.save(filename)

    def get_se2(self, num_rotations, pivot, reverse=False):
        '''
        Get SE2 rotations discretized into num_rotations angles counter-clockwise.
        Returns list (np.array) where each item is a flattened SE2 rotation matrix.
        '''
        rvecs = []
        for i in range(num_rotations):
            theta = i * 2 * np.pi / num_rotations
            theta = -theta if reverse else theta
            rmat = utils.get_image_transform(theta, (0, 0), pivot)
            rvec = rmat.reshape(-1)[:-1]
            rvecs.append(rvec)
        return np.array(rvecs, dtype=np.float32)

    def get_attention_heatmap(self, attention):
        """Given attention, get a human-readable heatmap.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html  
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode, red =
        hottest (highest attention values), green=medium, blue=lowest.

        Note: to see the grayscale only (which may be easier to interpret,
        actually...) save `vis_attention` just before applying the colormap.
        """
        # Options: cv2.COLORMAP_PLASMA, cv2.COLORMAP_JET, etc.
        #attention = tf.reshape(attention, (1, np.prod(attention.shape)))
        #attention = tf.nn.softmax(attention)
        vis_attention = np.float32(attention).reshape((320, 160))
        vis_attention = vis_attention - np.min(vis_attention)
        vis_attention = 255 * vis_attention / np.max(vis_attention)
        vis_attention = cv2.applyColorMap(np.uint8(vis_attention), cv2.COLORMAP_RAINBOW)
        vis_attention = cv2.cvtColor(vis_attention, cv2.COLOR_RGB2BGR)
        return vis_attention