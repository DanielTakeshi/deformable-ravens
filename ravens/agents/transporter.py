#!/usr/bin/env python

import os

import cv2
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from ravens.models import Attention, Transport, TransportGoal
from ravens import cameras
from ravens import utils
from ravens import tasks

import tensorflow as tf


class TransporterAgent:

    def __init__(self, name, task, num_rotations=24, crop_bef_q=True,
            use_goal_image=False, attn_no_targ=True):
        """Creates Transporter agent with attention and transport modules."""
        self.name = name
        self.task = task
        self.total_iter = 0
        self.crop_size = 64
        self.num_rotations = num_rotations
        self.pixel_size = 0.003125
        self.input_shape = (320, 160, 6)
        self.camera_config = cameras.RealSenseD415.CONFIG
        self.models_dir = os.path.join('checkpoints', self.name)
        #self.models_dir = os.path.join('/raid/seita/defs/checkpoints', self.name)
        #self.models_dir = os.path.join('/data/defs/checkpoints', self.name)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.crop_bef_q = crop_bef_q
        self.use_goal_image = use_goal_image
        self.attn_no_targ = attn_no_targ

        # TODO(daniel) Hacky. For bag-items-hard, pass in num_rotations since we use it?
        self.real_task = None

    def train(self, dataset, num_iter, writer):
        """Train on dataset for a specific number of iterations.

        Daniel: notice how little training data we use! One 'iteration' is
        simply one image and an associated action, drawn by (a) sampling
        demo, then (b) sampling time within it. We do heavy data
        augmentation, but it's still just one real image.

        If using a goal image, we use a different random_sample method that
        also picks the LAST image of that episode, which is assigned as the
        goal image. This would likely not work for super long-horizon tasks,
        but maybe; (Agarwal et al., NeurIPS 2016) in the PokeBot paper
        actually got something like this 'greedy-style' planning to work.
        Otherwise we might have to do something like (Nair et al., ICRA 2017)
        in the follow-up work where we feed in a target image for each time
        step, which would be the *next* image saved.

        For data augmentation with this goal image, I believe we should stack
        the current and goal image together, and THEN do augmentation. The
        perturb method will make sure placing pixels are preserved -- which
        for short-horizon environments usually means the goal image will
        contain most of the relevant information. When data augmenting, for
        both normal and goal-conditioned Transporters, the p1_theta
        (rotation) is the same, but pick points are correctly 'converted' to
        those appropriate for the augmented images.
        """
        for i in range(num_iter):
            if self.use_goal_image:
                obs, act, info, goal = dataset.random_sample(goal_images=True)
            else:
                obs, act, info = dataset.random_sample()

            # Get heightmap from RGB-D images.
            configs = act['camera_config']
            colormap, heightmap = self.get_heightmap(obs, configs)
            if self.use_goal_image:
                colormap_g, heightmap_g = self.get_heightmap(goal, configs)

            # Get training labels from data sample.
            pose0, pose1 = act['params']['pose0'], act['params']['pose1']
            p0_position, p0_rotation = pose0[0], pose0[1]
            p0 = utils.position_to_pixel(p0_position, self.bounds, self.pixel_size)
            p0_theta = -np.float32(p.getEulerFromQuaternion(p0_rotation)[2])
            p1_position, p1_rotation = pose1[0], pose1[1]
            p1 = utils.position_to_pixel(p1_position, self.bounds, self.pixel_size)
            p1_theta = -np.float32(p.getEulerFromQuaternion(p1_rotation)[2])
            p1_theta = p1_theta - p0_theta
            p0_theta = 0

            # Concatenate color with depth images.
            input_image = self.concatenate_c_h(colormap, heightmap)

            # If using goal image, stack _with_ input_image for data augmentation.
            if self.use_goal_image:
                goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
                input_image = np.concatenate((input_image, goal_image), axis=2)
                assert input_image.shape[2] == 12, input_image.shape

            # Do data augmentation (perturb rotation and translation).
            original_pixels = (p0, p1)
            input_image, pixels = utils.perturb(input_image, [p0, p1])
            p0, p1 = pixels

            # Optionally visualize images _after_ data agumentation.
            if False:
                self.visualize_images(p0, p0_theta, p1, p1_theta, original_pixels,
                        colormap=colormap, heightmap=heightmap,
                        colormap_g=colormap_g, heightmap_g=heightmap_g,
                        input_image=input_image, before_aug=False)

            # Compute Attention training loss.
            if self.attn_no_targ and self.use_goal_image:
                maxdim = int(input_image.shape[2] / 2)
                input_only = input_image[:, :, :maxdim]
                loss0 = self.attention_model.train(input_only, p0, p0_theta)
            else:
                loss0 = self.attention_model.train(input_image, p0, p0_theta)
            with writer.as_default():
                tf.summary.scalar('attention_loss', self.attention_model.metric.result(),
                    step=self.total_iter+i)

            # Compute Transport training loss.
            if isinstance(self.transport_model, Attention):
                loss1 = self.transport_model.train(input_image, p1, p1_theta)
            elif isinstance(self.transport_model, TransportGoal):
                half = int(input_image.shape[2] / 2)
                img_curr = input_image[:, :, :half]
                img_goal = input_image[:, :, half:]
                loss1 = self.transport_model.train(img_curr, img_goal, p0, p1, p1_theta)
            else:
                loss1 = self.transport_model.train(input_image, p0, p1, p1_theta)
            with writer.as_default():
                tf.summary.scalar('transport_loss', self.transport_model.metric.result(),
                    step=self.total_iter+i)

            print(f'Train Iter: {self.total_iter + i} Loss: {loss0:.4f} {loss1:.4f}')

        self.total_iter += num_iter
        self.save()

    def act(self, obs, info, debug_imgs=False, goal=None):
        """Run inference and return best action given visual observations.

        If goal-conditioning, provide `goal`. Both `obs` and `goal` have
        'color' and 'depth' keys, but `obs['color']` and `goal['color']` are
        of type list and np.array, respectively. This is different from
        training, above, where both `obs` and `goal` are sampled from the
        dataset class, which will load both as np.arrays. Here, the `goal` is
        still from dataset, but `obs` is from the environment stepping, which
        returns things in a list. Wrap an np.array(...) to get shapes:

        np.array(obs['color']) and goal['color']: (3, 480, 640, 3)
        np.array(obs['depth']) and goal['depth']: (3, 480, 640)
        """
        act = {'camera_config': self.camera_config, 'primitive': None}
        if not obs:
            return act

        # Get heightmap from RGB-D images.
        colormap, heightmap = self.get_heightmap(obs, self.camera_config)
        if goal is not None:
            colormap_g, heightmap_g = self.get_heightmap(goal, self.camera_config)

        # Concatenate color with depth images.
        input_image = self.concatenate_c_h(colormap, heightmap)

        # Make a goal image if needed, and for consistency stack with input.
        if self.use_goal_image:
            goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
            input_image = np.concatenate((input_image, goal_image), axis=2)
            assert input_image.shape[2] == 12, input_image.shape

        # Attention model forward pass.
        if self.attn_no_targ and self.use_goal_image:
            maxdim = int(input_image.shape[2] / 2)
            input_only = input_image[:, :, :maxdim]
            attention = self.attention_model.forward(input_only)
        else:
            attention = self.attention_model.forward(input_image)
        argmax = np.argmax(attention)
        argmax = np.unravel_index(argmax, shape=attention.shape)
        p0_pixel = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / attention.shape[2])

        # Transport model forward pass.
        if isinstance(self.transport_model, TransportGoal):
            half = int(input_image.shape[2] / 2)
            img_curr = input_image[:, :, :half]
            img_goal = input_image[:, :, half:]
            transport = self.transport_model.forward(img_curr, img_goal, p0_pixel)
        else:
            transport = self.transport_model.forward(input_image, p0_pixel)
        argmax = np.argmax(transport)
        argmax = np.unravel_index(argmax, shape=transport.shape)
        p1_pixel = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / transport.shape[2])

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

        # Daniel: determine the task stage if applicable. (AND if loading only)
        if self.task in ['bag-items-easy', 'bag-items-hard', 'bag-color-goal']:
            self._determine_task_stage(p0_pixel, p1_pixel)

        # Daniel: only change is potentially returning more info.
        if debug_imgs:
            # FWIW, attention (320,160,1), and already has softmax applied.
            # Then the attention heat map will return a (160,320,3) image.
            # The transport also has softmax, and is shape (320,160,num_rot).
            # Thus, t_heat is actually a LIST of (160,320,3) shaped images.
            # (For forward passes, we apply the softmax to the attention and
            # transport tensors; for training we don't because the TensorFlow
            # cross entropy loss assumes it's applied before the softmax.)
            a_heat = self.attention_model.get_attention_heatmap(attention)
            t_heat = self.transport_model.get_transport_heatmap(transport)
            extras = {
                'input_c': cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR),
                'attn_heat_bgr': a_heat,  # already converted to BGR
                'tran_heat_bgr': t_heat,  # already converted to BGR
                'tran_rot_argmax': argmax[2],
                'tran_p1_theta': p1_theta,
            }
            # Images by default should be vertically oriented. Can make
            # horizontal if we use .transpose(1,0,2).
            return act, extras
        else:
            return act

    #-------------------------------------------------------------------------
    # Helper Functions
    #-------------------------------------------------------------------------

    def concatenate_c_h(self, colormap, heightmap):
        """Concatenates color and height images to get a 6D image."""
        img = np.concatenate((colormap,
                              heightmap[..., None],
                              heightmap[..., None],
                              heightmap[..., None]), axis=2)
        assert img.shape == self.input_shape, img.shape
        return img

    def preprocess(self, image):
        """Pre-process images (subtract mean, divide by std)."""
        color_mean = 0.18877631
        depth_mean = 0.00509261
        color_std = 0.07276466
        depth_std = 0.00903967
        if image.shape[2] == 12:
            # This should only be for the Attention module in GCTN.
            assert self.use_goal_image
            image[:, :, 0:3] = (image[:, :, 0:3] / 255 - color_mean) / color_std
            image[:, :, 3:6] = (image[:, :, 3:6] - depth_mean) / depth_std
            image[:, :, 6:9] = (image[:, :, 6:9] / 255 - color_mean) / color_std
            image[:, :, 9:]  = (image[:, :, 9:] - depth_mean) / depth_std
        elif image.shape[2] == 6:
            # Transport-Goal calls processing separately for input and goal.
            image[:, :, :3] = (image[:, :, :3] / 255 - color_mean) / color_std
            image[:, :, 3:] = (image[:, :, 3:] - depth_mean) / depth_std
        else:
            raise ValueError(image.shape)
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
        """Load pre-trained models."""
        attention_fname = 'attention-ckpt-%d.h5' % num_iter
        transport_fname = 'transport-ckpt-%d.h5' % num_iter
        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname = os.path.join(self.models_dir, transport_fname)
        self.attention_model.load(attention_fname)
        self.transport_model.load(transport_fname)
        self.total_iter = num_iter

    def save(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        attention_fname = 'attention-ckpt-%d.h5' % self.total_iter
        transport_fname = 'transport-ckpt-%d.h5' % self.total_iter
        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname = os.path.join(self.models_dir, transport_fname)
        self.attention_model.save(attention_fname)
        self.transport_model.save(transport_fname)

    def _determine_task_stage(self, p0_pixel, p1_pixel):
        """Determines task stage for the bag-items tasks.

        Hacky solution, unfortunately, assumes we assigned task.env. Assumes that we
        have an actual `self.real_task` we can use; `self.task` is just a string.
        Currently working reasonably well for bag-items-easy. Note: see gt_state.py
        for the version that works for the gt_state baselines. Also, note that the
        self.real_task.task_stage is reset to 1 whenever we reset() so there's no
        need to deal with that logic here.

        (0) So far this is working reasonably well. Note that:
            - if object_mask[p0_pixel] == 38, then we're picking a cube.
            - for bag-items-hard, ID 39 is also a block (note: no cubes).
            - and if we end up putting both items into the bag, BUT we STILL end up
            with object_mask[p0_pixel] as gripping one of those items, we DO NOT go
            to task stage 3, which makes sense and is correct.

        (1) However, it will not be able to catch this failure:
            - putting two items in the bag.
            - but picking one item out of the bag, and inserting it to the target
            - then properly doing 'task stage 3' by pulling and inserting the bag
            with ONE item into the zone. However, the first item that was pulled out
            may be in the zone, leading to an 'undesirable' success.

        (2) Another potential failure that I observed:
            - FIRST action: sometimes the robot grips an item at the start, which is
            the right thing to do if the bag is already open. But we may grip the block
            EVEN THOUGH the pixel may not correspond to the block in the the segmentation
            mask. This happens when the pick point is just outside the block's pixels.
            Then this will result in a very slow pick and place since we follow task
            stage 1 parameters with slow movements and often get movej time outs if the
            block started far from the bag. In that case we can argue that we should have
            had a better pick point so that we correctly detect that we should move onto
            task stage 2. A better solution would be to check for any contact points
            after we grip, and test for contact with block IDs?
            - I ended up increasing the task stage 1 speed from 0.003 --> 0.004.

        (3) Of course, there is still a failure of picking a vertex instead of a bag bead
        during the bag pulling stage. AH, oh well.
        """
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

        # For bag-color-goal, it's fine to reset to task stage 1 if needed.
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

    #-------------------------------------------------------------------------
    # Visualization.
    #-------------------------------------------------------------------------

    def visualize_images(self, p0, p0_theta, p1, p1_theta, original_pixels,
            colormap, heightmap, colormap_g, heightmap_g, input_image, before_aug):
        """Daniel: code to debug and visualuze the image (including perturbed).

        The height maps will not be grayscale because of matplotlib's color
        scheme, I think. Using cv2.imwrite(..., heightmap) shows grayscale.
        """
        print(f'\nForward pass.')
        p0_theta_d = (180 / np.pi) * p0_theta
        p1_theta_d = (180 / np.pi) * p1_theta
        heightmap = heightmap / np.max(heightmap) * 255
        heightmap_g = heightmap_g / np.max(heightmap_g) * 255

        plt.subplots(1, 6, figsize=(12,4))
        plt.subplot(1,6,1)
        plt.imshow(colormap)
        plt.subplot(1,6,2)
        plt.imshow(heightmap)
        plt.subplot(1,6,3)
        plt.imshow(colormap_g)
        plt.subplot(1,6,4)
        plt.imshow(heightmap_g)
        plt.subplot(1,6,5)
        plt.imshow(np.array(input_image[:,:,0:3]).astype(np.uint8))
        plt.subplot(1,6,6)
        plt.imshow(np.array(input_image[:,:,6:9]).astype(np.uint8))

        op_0, op_1 = original_pixels
        title = f'Before Data Aug: ' \
                f'Pick: ({op_0}, {p0_theta:0.2f}) ' \
                f'Place: ({op_1}, {p1_theta:0.2f}={p1_theta_d:0.2f})\n' \
                f'After Data Aug: ' \
                f'Pick: ({p0}) ' \
                f'Place: ({p1})'
        plt.suptitle(title, fontsize=15)
        plt.tight_layout()
        plt.show()


class OriginalTransporterAgent(TransporterAgent):
    """
    The official Transporter agent tested in the paper. Added num_rotations and
    crop_bef_q as arguments. Default arguments are 24 (though this was later
    turned to 36 for Andy's paper) and to crop to get kernels _before_ the query.
    """

    def __init__(self, name, task, num_rotations=24, crop_bef_q=True):
        super().__init__(name, task, num_rotations, crop_bef_q=crop_bef_q)

        self.attention_model = Attention(input_shape=self.input_shape,
                                         num_rotations=1,
                                         preprocess=self.preprocess)
        self.transport_model = Transport(input_shape=self.input_shape,
                                         num_rotations=self.num_rotations,
                                         crop_size=self.crop_size,
                                         preprocess=self.preprocess,
                                         per_pixel_loss=False,
                                         crop_bef_q=self.crop_bef_q)


class GoalTransporterAgent(TransporterAgent):
    """
    Goal-conditioned Transporter agent where we pass the goal image through another FCN,
    and then combine the resulting features with the pick and placing networks for better
    goal-conditioning. This uses our new `TransportGoal` architecture. We don't stack the
    input and target images, so we can directly use `self.input_shape` for both modules.

    NOTE(daniel) from March 2023: this is Transporter-Goal-Split in the paper.
    """

    def __init__(self, name, task, num_rotations=24):
        # (Oct 26) set attn_no_targ=False, and that should be all we need along w/shape ...
        super().__init__(name, task, num_rotations, use_goal_image=True, attn_no_targ=False)

        # (Oct 26) Stack the goal image for the Attention module -- model cannot pick properly otherwise.
        a_shape = (self.input_shape[0], self.input_shape[1], int(self.input_shape[2] * 2))

        self.attention_model = Attention(input_shape=a_shape,
                                         num_rotations=1,
                                         preprocess=self.preprocess)
        self.transport_model = TransportGoal(input_shape=self.input_shape,
                                             num_rotations=self.num_rotations,
                                             crop_size=self.crop_size,
                                             preprocess=self.preprocess)


class GoalNaiveTransporterAgent(TransporterAgent):
    """Same as super naive, except the target image isn't given to the Attention module.

    NOTE: when I trained these before mid-October, we actually did not set a crop_bef_q value,
    hence we were using the default of True, which matches the CoRL paper but is a little different
    from the logic of the Transporter-Goal model, hence I'm going to use crop_bef_q=False explicitly.
    However, this means we can't actually use the models trained in test time unless we also change
    this setting. It won't throw an error (number of parameters in Transport model is the same) but
    it means the logic is bad; the filters will be applied on the 'wrong' images.

    NOTE(daniel) from March 2023: we did not report results for this in the paper. Actually
    it would not make sense if the target image were not given to the Attention module.
    """

    def __init__(self, name, task, num_rotations=24):
        super().__init__(name, task, num_rotations, use_goal_image=True, attn_no_targ=True)

        # We stack the goal image for the Transport module. (Oct 26: do not use, use SuperNaive instead)
        t_shape = (self.input_shape[0], self.input_shape[1], int(self.input_shape[2] * 2))

        self.attention_model = Attention(input_shape=self.input_shape,
                                         num_rotations=1,
                                         preprocess=self.preprocess)
        self.transport_model = Transport(input_shape=t_shape,
                                         num_rotations=self.num_rotations,
                                         crop_size=self.crop_size,
                                         preprocess=self.preprocess,
                                         per_pixel_loss=False,
                                         crop_bef_q=False,  # NOTE: see docs above.
                                         use_goal_image=True)


class GoalSuperNaiveTransporterAgent(TransporterAgent):
    """
    A super naive goal-conditioned Transporter agent, where the image input for
    both attention and transport is just the current and goal stacked channel-wise.

    NOTE: it would probably be better to set crop_bef_q=False to be closer to what
    Transporter-Goal does, but I ended up training without initially, and only realized
    this after the fact, hence keeping crop_bef_q unspecified for now, which means it
    defaults to the True setting in Transport() class.

    NOTE(daniel) from March 2023: this is Transporter-Goal-Stack in the paper.
    """

    def __init__(self, name, task, num_rotations=24):
        super().__init__(name, task, num_rotations, use_goal_image=True, attn_no_targ=False)

        # We stack the goal image for both modules.
        in_shape = (self.input_shape[0], self.input_shape[1], int(self.input_shape[2] * 2))

        self.attention_model = Attention(input_shape=in_shape,
                                         num_rotations=1,
                                         preprocess=self.preprocess)
        self.transport_model = Transport(input_shape=in_shape,
                                         num_rotations=self.num_rotations,
                                         crop_size=self.crop_size,
                                         preprocess=self.preprocess,
                                         per_pixel_loss=False,  # NOTE: see docs above.
                                         use_goal_image=True)


class NoTransportTransporterAgent(TransporterAgent):
    """
    For this baseline, the transport model is also an attention model, so it
    does not get a pick-conditioned input. To handle rotations, provide it
    with num_rotations (while keeping the picking network with one rotation).
    """

    def __init__(self, name, task):
        super().__init__(name, task)

        self.attention_model = Attention(input_shape=self.input_shape,
                                         num_rotations=1,
                                         preprocess=self.preprocess)
        self.transport_model = Attention(input_shape=self.input_shape,
                                         num_rotations=self.num_rotations,
                                         preprocess=self.preprocess)


class PerPixelLossTransporterAgent(TransporterAgent):

    def __init__(self, name, task):
        super().__init__(name, task)

        self.attention_model = Attention(input_shape=self.input_shape,
                                         num_rotations=1,
                                         preprocess=self.preprocess)
        self.transport_model = Transport(input_shape=self.input_shape,
                                         num_rotations=self.num_rotations,
                                         crop_size=self.crop_size,
                                         preprocess=self.preprocess,
                                         per_pixel_loss=True)
