#!/usr/bin/env python

import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from ravens.models import ResNet43_8s
from ravens import utils


class Transport:
    """Daniel: Transporter for the placing module.

    By default, don't use per-pixel loss, meaning that pixels collectively
    define the set of possible classes, and we just pick one of them. Per
    pixel means each pixel has positive and negative (and maybe a third
    neutral) label. Also, usually rotations=24 and crop_size=64.

    Also by default, we crop and then pass the input to the query network. We
    could also pass the full image to the query network, and then crop it.

    Note the *two* FCNs here (in0, out0, in1, out1) because this Transport
    model has two streams. Then `self.model` gets both in0 and in1 as input.
    Shapes below assume we get crops and then pass them to the query net.

    Image-to-image   (phi) = {in,out}0
    Kernel-to-kernel (psi) = {in,out}1
    input_shape: (384, 224, 6)
    kernel_shape: (64, 64, 6)
    in0: Tensor("input_2:0", shape=(None, 384, 224, 6), dtype=float32)
    in1: Tensor("input_3:0", shape=(None, 64, 64, 6), dtype=float32)
    out0: Tensor("add_31/Identity:0", shape=(None, 384, 224, 3), dtype=float32)
    out1: Tensor("add_47/Identity:0", shape=(None, 64, 64, 3), dtype=float32)

    The batch size for the forward pass is the number of rotations, by
    default 24 (they changed to 36 later).
    """

    def __init__(self, input_shape, num_rotations, crop_size, preprocess,
                 per_pixel_loss=False, crop_bef_q=True, use_goal_image=False):
        self.num_rotations = num_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.per_pixel_loss = per_pixel_loss
        self.crop_bef_q = crop_bef_q
        self.use_goal_image = use_goal_image

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        input_shape = np.array(input_shape)
        input_shape[0:2] += self.pad_size * 2
        input_shape = tuple(input_shape)

        kernel_shape = (self.crop_size, self.crop_size, input_shape[2])
        output_dim = 6 if self.per_pixel_loss else 3

        # 2 fully convolutional ResNets [Daniel: I think 43 layers and stride 8]
        in0, out0 = ResNet43_8s(input_shape, output_dim, prefix='s0_')
        if self.crop_bef_q:
            in1, out1 = ResNet43_8s(kernel_shape, 3, prefix='s1_')
        else:
            in1, out1 = ResNet43_8s(input_shape, output_dim, prefix='s1_')

        self.model = tf.keras.Model(inputs=[in0, in1], outputs=[out0, out1])
        self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.metric = tf.keras.metrics.Mean(name='transport_loss')

    def forward(self, in_img, p, apply_softmax=True):
        """Forward pass.

        Daniel: dissecting this a bit more since it's the key technical
        contribution. Relevant shapes and info:

            in_img: (320,160,6)
            p: integer pixels on in_img, e.g., [158, 30]
            self.padding: [[32,32],[32,32],0,0]], with shape (3,2)

        How does the cross-convolution work? We get the output from BOTH
        streams of the network. Then, the output for one stream is the set of
        convolutional kernels for the other. Call tf.nn.convolution. That's
        it, and the entire operation is differentiable, so that gradients
        will apply to both of the two Transport FCN streams.
        """
        input_data = np.pad(in_img, self.padding, mode='constant')      # (384,224,6)
        input_data = self.preprocess(input_data)                        # (384,224,6)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)                    # (1,384,224,6)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,384,224,6)
        pivot = np.array([p[1], p[0]]) + self.pad_size
        rvecs = self.get_se2(self.num_rotations, pivot)

        if self.crop_bef_q:
            # Rotate crop, pass crop of (24,64,64,6) to query to get kernel of (24,64,64,3).
            crop = tf.convert_to_tensor(input_data.copy(), dtype=tf.float32)    # (1,384,224,6)
            crop = tf.repeat(crop, repeats=self.num_rotations, axis=0)          # (24,384,224,6)
            crop = tfa.image.transform(crop, rvecs, interpolation='NEAREST')    # (24,384,224,6)
            crop = crop[:,
                        p[0]:(p[0] + self.crop_size),
                        p[1]:(p[1] + self.crop_size),
                        :]
            logits, kernel = self.model([in_tensor, crop])
        else:
            # Pass `in_tensor` twice, get crop from `kernel_before_crop` (not `input_data`).
            logits, kernel_before_crop = self.model([in_tensor, in_tensor])
            crop = tf.identity(kernel_before_crop)                              # (1,384,224,3)
            crop = tf.repeat(crop, repeats=self.num_rotations, axis=0)          # (24,384,224,3)
            crop = tfa.image.transform(crop, rvecs, interpolation='NEAREST')    # (24,384,224,3)
            kernel = crop[:,
                          p[0]:(p[0] + self.crop_size),
                          p[1]:(p[1] + self.crop_size),
                          :]

        # Cross-convolve crop. Here, pad kernel: (24,64,64,3) --> (65,65,3,24).
        kernel_paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        kernel = tf.pad(kernel, kernel_paddings, mode='CONSTANT')
        kernel = tf.transpose(kernel, [1, 2, 3, 0])

        if self.per_pixel_loss:
            output0 = tf.nn.convolution(logits[..., :3], kernel, data_format="NHWC")
            output1 = tf.nn.convolution(logits[..., 3:], kernel, data_format="NHWC")
            output = tf.concat((output0, output1), axis=0)
            output = tf.transpose(output, [1, 2, 3, 0])
        else:
            output = tf.nn.convolution(logits, kernel, data_format="NHWC")
        output = (1 / (self.crop_size**2)) * output

        if apply_softmax:
            output_shape = output.shape
            if self.per_pixel_loss:
                output = tf.reshape(output, (np.prod(output.shape[:-1]), 2))
            else:
                output = tf.reshape(output, (1, np.prod(output.shape)))
            output = tf.nn.softmax(output)
            if self.per_pixel_loss:
                output = np.float32(output[:, 1]).reshape(output_shape[:-1])
            else:
                output = np.float32(output).reshape(output_shape[1:])

        # Daniel: visualize crops and kernels (latter should be testing only).
        # We also have code (called later) to visualize full `output`.
        #self.visualize_images(p, in_img, input_data, crop)
        #self.visualize_transport(p, in_img, input_data, crop, kernel)
        #self.visualize_logits(logits)

        return output

    def train(self, in_img, p, q, theta):
        """Transport pixel p to pixel q.

          Args:
            input:
            depth_image:
            p: pixel (y, x)
            q: pixel (y, x)
          Returns:
            A `Tensor`. Has the same type as `input`.

        Daniel: the `in_img` will include the color and depth. Much is
        similar to the attention model if we're not using the per-pixel loss:
        (a) forward pass, (b) get angle discretizations [though we set only 1
        rotation for the picking model], (c) make the label consider
        rotations in the last axis, but only provide the label to one single
        (pixel,rotation) combination, (d) follow same exact steps for the
        non-per pixel loss otherwise. The output reshaping to (1, ...) is
        done in the attention model forward pass, but not in the transport
        forward pass. Note the `1` meaning a batch size of 1.
        """
        self.metric.reset_states()
        with tf.GradientTape() as tape:
            output = self.forward(in_img, p, apply_softmax=False)

            itheta = theta / (2 * np.pi / self.num_rotations)
            itheta = np.int32(np.round(itheta)) % self.num_rotations

            label_size = in_img.shape[:2] + (self.num_rotations,)
            label = np.zeros(label_size)
            label[q[0], q[1], itheta] = 1

            if self.per_pixel_loss:
                sampling = True  # sampling negatives seems to converge faster
                if sampling:
                    num_samples = 100
                    inegative = utils.sample_distribution(1 - label, num_samples)
                    inegative = [np.ravel_multi_index(i, label.shape) for i in inegative]
                    ipositive = np.ravel_multi_index([q[0], q[1], itheta], label.shape)
                    output = tf.reshape(output, (-1, 2))
                    output_samples = ()
                    for i in inegative:
                        output_samples += (tf.reshape(output[i, :], (1, 2)),)
                    output_samples += (tf.reshape(output[ipositive, :], (1, 2)),)
                    output = tf.concat(output_samples, axis=0)
                    label = np.int32([0] * num_samples + [1])[..., None]
                    label = np.hstack((1 - label, label))
                    weights = np.ones(label.shape[0])
                    weights[:num_samples] = 1./num_samples
                    weights = weights / np.sum(weights)

                else:
                    ipositive = np.ravel_multi_index([q[0], q[1], itheta], label.shape)
                    output = tf.reshape(output, (-1, 2))
                    label = np.int32(np.reshape(label, (int(np.prod(label.shape)), 1)))
                    label = np.hstack((1 - label, label))
                    weights = np.ones(label.shape[0]) * 0.0025  # magic constant
                    weights[ipositive] = 1

                label = tf.convert_to_tensor(label, dtype=tf.int32)
                weights = tf.convert_to_tensor(weights, dtype=tf.float32)
                loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
                loss = tf.reduce_mean(loss * weights)

            else:
                label = label.reshape(1, np.prod(label.shape))
                label = tf.convert_to_tensor(label, dtype=tf.float32)
                output = tf.reshape(output, (1, np.prod(output.shape)))
                loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
                loss = tf.reduce_mean(loss)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grad, self.model.trainable_variables))

        self.metric(loss)

        return np.float32(loss)

    def get_se2(self, num_rotations, pivot):
        '''
        Get SE2 rotations discretized into num_rotations angles counter-clockwise.
        '''
        rvecs = []
        for i in range(num_rotations):
            theta = i * 2 * np.pi / num_rotations
            rmat = utils.get_image_transform(theta, (0, 0), pivot)
            rvec = rmat.reshape(-1)[:-1]
            rvecs.append(rvec)
        return np.array(rvecs, dtype=np.float32)

    def save(self, fname):
        self.model.save(fname)

    def load(self, fname):
        self.model.load_weights(fname)

    #-------------------------------------------------------------------------
    # Visualization.
    #-------------------------------------------------------------------------

    def visualize_images(self, p, in_img, input_data, crop):
        def get_itheta(theta):
            itheta = theta / (2 * np.pi / self.num_rotations)
            return np.int32(np.round(itheta)) % self.num_rotations

        plt.subplot(1, 3, 1)
        plt.title(f'Perturbed', fontsize=15)
        plt.imshow(np.array(in_img[:, :, :3]).astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.title(f'Process/Pad', fontsize=15)
        plt.imshow(input_data[0, :, :, :3])
        plt.subplot(1, 3, 3)
        # Let's stack two crops together.
        theta1 = 0.0
        theta2 = 90.0
        itheta1 = get_itheta(theta1)
        itheta2 = get_itheta(theta2)
        crop1 = crop[itheta1, :, :, :3]
        crop2 = crop[itheta2, :, :, :3]
        barrier = np.ones_like(crop1)
        barrier = barrier[:4, :, :] # white barrier of 4 pixels
        stacked = np.concatenate((crop1, barrier, crop2), axis=0)
        plt.imshow(stacked)
        plt.title(f'{theta1}, {theta2}', fontsize=15)
        plt.suptitle(f'pick: {p}', fontsize=15)
        plt.tight_layout()
        plt.show()
        #plt.savefig('viz.png')

    def visualize_transport(self, p, in_img, input_data, crop, kernel):
        """Like the attention map, let's visualize the transport data from a
        trained model.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode,
        red=hottest (highest attention values), green=medium, blue=lowest.

        See also:
        https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.subplot.html

        crop.shape: (24,64,64,6)
        kernel.shape = (65,65,3,24)
        """
        def colorize(img):
            # I don't think we have to convert to BGR here...
            img = img - np.min(img)
            img = 255 * img / np.max(img)
            img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_RAINBOW)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        kernel = (tf.transpose(kernel, [3, 0, 1, 2])).numpy()

        # Top two rows: crops from processed RGBD. Bottom two: output from FCN.
        nrows = 4
        ncols = 12
        assert self.num_rotations == nrows * (ncols / 2)
        idx = 0
        fig, ax = plt.subplots(nrows, ncols, figsize=(12,6))
        for _ in range(nrows):
            for _ in range(ncols):
                plt.subplot(nrows, ncols, idx+1)
                plt.axis('off')  # Ah, you need to put this here ...
                if idx < self.num_rotations:
                    plt.imshow(crop[idx, :, :, :3])
                else:
                    # Offset because idx goes from 0 to (rotations * 2) - 1.
                    _idx = idx - self.num_rotations
                    processed = colorize(img=kernel[_idx, :, :, :])
                    plt.imshow(processed)
                idx += 1
        plt.tight_layout()
        plt.show()

    def visualize_logits(self, logits):
        """Given logits (BEFORE tf.nn.convolution), get heatmap.

        Here we apply a softmax to make it more human-readable. However, the
        tf.nn.convolution with the learned kernels happens without a softmax
        on the logits.
        """
        original_shape = logits.shape
        logits = tf.reshape(logits, (1, np.prod(original_shape)))
        logits = tf.nn.softmax(logits)
        vis_transport = np.float32(logits).reshape(original_shape)
        vis_transport = vis_transport[0]
        vis_transport = vis_transport - np.min(vis_transport)
        vis_transport = 255 * vis_transport / np.max(vis_transport)
        vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)
        # Only if we're saving with cv2.imwrite
        #vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
        plt.subplot(1, 1, 1)
        plt.title(f'Logits', fontsize=15)
        plt.imshow(vis_transport)
        plt.tight_layout()
        plt.show()

    def get_transport_heatmap(self, transport):
        """Given transport output, get a human-readable heatmap.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode, red =
        hottest (highest attention values), green=medium, blue=lowest.
        """
        # Options: cv2.COLORMAP_PLASMA, cv2.COLORMAP_JET, etc.
        #transport = tf.reshape(transport, (1, np.prod(transport.shape)))
        #transport = tf.nn.softmax(transport)
        assert transport.shape == (320, 160, self.num_rotations), transport.shape
        vis_images = []
        for idx in range(self.num_rotations):
            t_img = transport[:, :, idx]
            vis_transport = np.float32(t_img)
            vis_transport = vis_transport - np.min(vis_transport)
            vis_transport = 255 * vis_transport / np.max(vis_transport)
            vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)
            vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
            vis_images.append(vis_transport)
        return vis_images
