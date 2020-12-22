#!/usr/bin/env python
"""Strictly for loading agents to inspect. Based on `main.py`."""

import datetime
import os
import time
import argparse
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os.path import join
from ravens import Dataset, Environment, cameras, agents, tasks
from ravens import utils as U

# Of critical importance! See the top of main.py for details.
MAX_ORDER = 4

# See Task().
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])


def goal_similarity(obs, goal):
    """For goal-conditioning, measure how close current image is to goal.

    Metrics: L2 and SSIM for now. The `obs` and `goal` should be of the same
    format as in rollout(), where they have color/depth keys, with 3 camera
    viewpoints. However, `obs` will be a list and `goal a np.array. For the
    pose metrics, use the task reward.
    """
    # Requires pip install scikit-image
    from skimage.metrics import structural_similarity

    colormap_o, _ = get_heightmap(obs=obs)
    colormap_g, _ = get_heightmap(obs=goal)
    L2 = np.linalg.norm(colormap_o - colormap_g) / np.prod(colormap_o.shape)
    SSIM = structural_similarity(colormap_o, colormap_g, multichannel=True)
    metrics = {}
    metrics['L2'] = round(L2, 4)
    metrics['SSIM'] = round(SSIM, 4)
    return metrics


def get_heightmap(obs):
    """Reconstruct orthographic heightmaps with segmentation masks.

    Here, `obs` could be current or goal, either will work.
    See transporter.py, regression.py, task.py, dummy.py, and dataset.py.
    We use this pattern quite a lot. Copy from transporter.py version.
    """
    heightmaps, colormaps = U.reconstruct_heightmaps(
        obs['color'], obs['depth'], CAMERA_CONFIG, BOUNDS, PIXEL_SIZE)
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


def load(path, iepisode, field):
    """Adapted from `dataset.py` so we can sample goal images. Just including
    some logic to extract the episode automatically based on the index
    `iepisode`, so we don't need to know the length in advance.
    """
    field_path = os.path.join(path, field)
    data_list = [os.path.join(field_path, x) for x in os.listdir(field_path)]
    fname = [x for x in data_list if f'{iepisode:06d}' in x]
    assert len(fname) == 1, fname
    fname = fname[0]
    return pickle.load(open(fname, 'rb'))


def debug_time_step(t, epidx, obs, act, extras, goal=None):
    """Save images and other stuff from time `t` in episode `epidx`."""
    pth = 'tmp'
    tt = str(t).zfill(2)

    # Convert from BGR to RGB to match what we see in the GUI.
    def save(fname, c_img):
        cv2.imwrite(fname, img=cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB))

    # Save current color images from camera angles and the fused version.
    for img_idx, c_img in enumerate(obs['color']):
        fname = join(pth, f'ep_{epidx}_t{tt}_cimg_{img_idx}.png')
        save(fname, c_img)
    colormap_o, _ = get_heightmap(obs=obs)
    fname = join(pth, f'ep_{epidx}_t{tt}_cimg_fused.png')
    save(fname, colormap_o)

    # (If applicable) save the goal color images.
    if (goal is not None) and t == 1:
        for img_idx, c_img in enumerate(goal['color']):
            fname = join(pth, f'ep_{epidx}_t{tt}_cimg_{img_idx}_goal.png')
            save(fname, c_img)
        colormap_g, _ = get_heightmap(obs=goal)
        fname = join(pth, f'ep_{epidx}_t{tt}_cimg_fused_goal.png')
        save(fname, colormap_g)

    # Print the action.
    pose0 = act['params']['pose0']
    pose1 = act['params']['pose1']
    print(f"  pose0, pose1: {U.round_pose(pose0)}, {U.round_pose(pose1)}")

    # Attention. (Well, attn_input.png is also input to Transport...)
    fname1 = join(pth, f'ep_{epidx}_t{tt}_attn_input.png')
    fname2 = join(pth, f'ep_{epidx}_t{tt}_attn_heat_bgr.png')
    cv2.imwrite(fname1, extras['input_c'])
    cv2.imwrite(fname2, extras['attn_heat_bgr'])

    # Transport
    for idx, tran_heat in enumerate(extras['tran_heat_bgr']):
        idxstr = str(idx).zfill(2)
        fname = join(pth, f'ep_{epidx}_t{tt}_tran_rot_{idxstr}.png')
        if idx == extras['tran_rot_argmax']:
            fname = fname.replace('.png', '_rot_chosen.png')
        cv2.imwrite(fname, tran_heat)


def rollout(agent, env, task, goal_conditioned, args, num_finished, debug=False):
    """Standard gym environment rollout.

    Adding more debugging options (enable with debug=True), such as printing
    the pose and saving the images and heatmaps. We can also run `dataset.py`
    and see goal images in the `goals_out` directory.

    :goal_conditioned: a boolean to check if we have goal-conditioning.
    :num_finished: to track how many episodes we have finished. Ignores any
        episodes drawn and then discarded due to initial states that were
        already done. Also used to sample the goal states for
        goal-conditioned policies. We have a fixed number of testing episodes
        (characterized by goal images), so `num_finished` is the identifier.

    Returns `t` to track episode length. Update (21 Aug 2020): also returns
    last_stuff=(obs,info), consistent with main.py and generate_goals.py.

    (13 Oct 2020): fixing so that we will always append stuff in the episode
    list for gt_state agents. The problem is that the first time step (start_t=1)
    wasn't saving because len(obs) = 0, but in gt_state we actually want to save.
    Otherwise, a length 1 episode will have len(episode)==0 later. It's not a huge
    deal because we still save the final info correctly, so that we can report
    correct stats, but it helps to have the initial info because that gives us the
    deltas over the starting state.
    """
    if debug:
        if not os.path.exists('tmp/'):
            os.makedirs('tmp/')
        print('')
    start_t = 0
    if args.agent in ['gt_state', 'gt_state_2_step']:
        start_t = 1
    episode = []
    total_reward = 0

    # Before task.reset(), need goal info for goal episode at idx `num_finished`.
    if goal_conditioned:
        task.goal_cond_testing = True
        path = os.path.join('goals', args.task)
        goal = {}
        goal['color'] = load(path, num_finished, 'last_color')
        goal['depth'] = load(path, num_finished, 'last_depth')
        goal['info'] = load(path, num_finished, 'last_info')

    goal_imgs = goal if goal_conditioned else None

    # Reset env and call task.reset(), len(obs)=0 but info will have stuff for gt_state.
    if goal_conditioned:
        obs = env.reset(task, last_info=goal['info'])
    else:
        obs = env.reset(task)
    info = env.info

    for t in range(start_t, task.max_steps):
        if debug and t > 0:
            act, extras = agent.act(obs, info, goal=goal_imgs, debug_imgs=True)
        else:
            act = agent.act(obs, info, goal=goal_imgs)

        # Optional debugging to save images, etc. Do before we get new obs.
        if debug and 'params' in act:
            debug_time_step(t, num_finished, obs, act, extras, goal=goal_imgs)

        # (13 Oct 2020) Ah, if gt_state, we won't save at start_t=1, so let's fix that!
        if (len(obs) > 0 and act['primitive']) or (args.agent in ['gt_state', 'gt_state_2_step']):
            episode.append((act, info)) # don't save obs
        (obs, reward, done, info) = env.step(act)

        # If goal-conditioning, additionally compute image-based metrics.
        if goal_conditioned and ('color' in obs and 'depth' in obs):
            info['image_metrics'] = goal_similarity(obs, goal_imgs)
        else:
            info['image_metrics'] = None

        if debug:
            print('  {}/{}, rew: {:0.3f}, len(epis): {}, act: {}, info: {}'.format(t,
                    task.max_steps, reward, len(episode), act['primitive'], info['extras']))
            if goal_conditioned:
                print('  goal-conditioning image metrics: {}'.format(info['image_metrics']))

        total_reward += reward
        last_obs_info = (obs, info)
        if done:
            break
    return total_reward, episode, t, last_obs_info


def is_goal_conditioned(args):
    """
    Be careful with checking this condition. See `generate_goals.py`. Here,
    though, we check the task name and as an extra safety measure, check that
    the agent is also named with 'goal'.

    Update: all right, let's modify this to incorpoate gt_state w/out too much
    extra work. :(
    """
    goal_tasks = ['insertion-goal', 'cable-shape-notarget', 'cable-line-notarget',
            'cloth-flat-notarget', 'bag-color-goal']
    goal_task = (args.task in goal_tasks)
    if goal_task:
        assert 'goal' in args.agent or 'gt_state' in args.agent, \
            'Agent should be a goal-based agent, or gt_state agent.'
    return goal_task


def ignore_this_demo(args, reward, t, last_extras):
    """In some cases, we should filter out demonstrations.

    Filter for if t == 0, which means the initial state was a success, and
    also if we have exit_gracefully, which means for the bag-items tasks, it
    may not have had visible item(s) at the start, for some reason.
    """
    ignore = (t == 0)
    if 'exit_gracefully' in last_extras:
        assert last_extras['exit_gracefully']
        return True
    return ignore


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',            default='0')
    parser.add_argument('--disp',           action='store_true')
    parser.add_argument('--task',           default='hanoi')
    parser.add_argument('--agent',          default='transporter')
    parser.add_argument('--num_demos',      default=1000, type=int)
    parser.add_argument('--train_run',      default=0, type=int)
    parser.add_argument('--num_test_eps',   default=20, type=int)
    parser.add_argument('--num_rots',       default=24, type=int,
        help='Transporter rotations used from the trained model, usually 24')
    parser.add_argument('--num_rots_inf',   default=24, type=int,
        help='Transporter rotations we want FOR INFERENCE time; it can be 1')
    parser.add_argument('--hz',             default=240.0, type=float)
    parser.add_argument('--crop_bef_q',     default=0, type=int, help='CoRL paper used 1')
    parser.add_argument('--gpu_mem_limit',  default=None)
    parser.add_argument('--subsamp_g',      action='store_true')
    args = parser.parse_args()

    # Configure which GPU to use.
    cfg = tf.config.experimental
    gpus = cfg.list_physical_devices('GPU')
    if len(gpus) == 0:
        print('No GPUs detected. Running with CPU.')
    else:
        cfg.set_visible_devices(gpus[int(args.gpu)], 'GPU')

    # Configure how much GPU to use.
    if args.gpu_mem_limit is not None:
        MEM_LIMIT = int(1024 * float(args.gpu_mem_limit))
        print(args.gpu_mem_limit)
        dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=MEM_LIMIT)]
        cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

    # Initialize task, set to 'test,' but I think this only matters for kitting.
    task = tasks.names[args.task]()
    task.mode = 'test'

    # Evaluate on saved snapshots. Go backwards to get better results first.
    snapshot_itrs = [i*2000 for i in range(1,10+1)]  # Do 10 snapshots to save on compute.
    snapshot_itrs = snapshot_itrs[::-1]
    if not os.path.exists('test_results'):
        os.makedirs('test_results')

    # Make environment once, due to issues with deformables + multiple calls.
    env = Environment(args.disp, hz=args.hz)

    # Check if it's goal-conditioned.
    goal_conditioned = is_goal_conditioned(args)

    for snapshot_itr in snapshot_itrs:
        # Set random seeds, so different snapshots test on same starting states.
        tf.random.set_seed(args.train_run)
        np.random.seed(args.train_run)

        # Set the beginning of the agent name.
        name = f'{args.task}-{args.agent}-{args.num_demos}-{args.train_run}'

        # Initialize agent and load from snapshot. NOTE: main difference from
        # main.py is to use num_rots_inf (not args.num_rots) for inference time.
        # Also, `self.name` must match what's in main.py, to load correct weights.
        if args.agent == 'transporter':
            name = f'{name}-rots-{args.num_rots}-crop_bef_q-{args.crop_bef_q}'
            agent = agents.names[args.agent](name,
                                             args.task,
                                             num_rotations=args.num_rots_inf,
                                             crop_bef_q=(args.crop_bef_q == 1))
        elif 'transporter-goal' in args.agent:
            assert goal_conditioned
            name = f'{name}-rots-{args.num_rots}'
            if args.subsamp_g:
                name += '-sub_g'
            else:
                name += '-fin_g'
            agent = agents.names[args.agent](name,
                                             args.task,
                                             num_rotations=args.num_rots_inf)
        elif 'gt_state' in args.agent:
            agent = agents.names[args.agent](name,
                                             args.task,
                                             one_rot_inf=(args.num_rots_inf==1),
                                             goal_conditioned=goal_conditioned)
        else:
            agent = agents.names[args.agent](name, args.task)

        agent.load(snapshot_itr)
        print(f'\nFinished loading snapshot: {snapshot_itr}, for: {name}.')

        # Hacky. Works for transporter and gt-state(2step) agents.
        agent.real_task = task

        # Evaluate agent. Save as list of (iter, episode_list, results(dict)).
        # List `episode_list` has all the `info`s BEFORE the last one (gives
        # starting state material), and the last one is `results['final_info']`.
        performance = []
        episode = 0
        finished = 0

        while finished < args.num_test_eps:
            seed = 10**MAX_ORDER + episode
            np.random.seed(seed)
            total_reward, episode_list, length, last_obs_info = rollout(
                    agent, env, task, goal_conditioned, args, num_finished=finished)
            _, info = last_obs_info  # ignore obs
            last_extras = info['extras']

            if ignore_this_demo(args, total_reward, t=length, last_extras=last_extras):
                print(f'  Ignoring demo, {last_extras}, not counting episode {episode}')
            else:
                result = {'reward': total_reward, 'length': length}
                result['final_info'] = info['extras']
                if goal_conditioned:
                    result['image_metrics'] = info['image_metrics']
                print(f'  Test (seed {seed}): {finished}. Results: {result}')
                performance.append((agent.total_iter, episode_list, result))
                finished += 1
            episode += 1

        # Save results.
        ss = str(snapshot_itr).zfill(5)
        rots_inf = str(args.num_rots_inf).zfill(2)
        base1 = f'{name}-rotsinf-{rots_inf}'
        base2 = f'snapshot-{ss}-eps-{args.num_test_eps}.pkl'
        head = os.path.join('test_results', base1)
        if not os.path.exists(head):
            os.makedirs(head)
        fpath = os.path.join(head, base2)
        with open(fpath, 'wb') as fh:
            pickle.dump(performance, fh)
