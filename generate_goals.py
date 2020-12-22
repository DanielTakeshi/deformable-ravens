#!/usr/bin/env python
"""For generating goals. Run like this:

python generate_goals.py --disp --hz=240 --task=insertion-goal --num_goals=20

where the hz and task should be selected appropriately. Probably ~20 goals is
OK. See: https://github.com/DanielTakeshi/pybullet-def-envs/pull/15 We should
put goal-based information (e.g., object poses, etc.) in `info` for now.
"""
import os
import cv2
import argparse
import numpy as np
from ravens import Dataset, Environment, agents, tasks

# Of critical importance! See main.py for documentation.
MAX_ORDER = 5


def rollout(agent, env, task):
    """Standard gym environment rollout, following as in main.py."""
    episode = []
    total_reward = 0
    obs = env.reset(task)
    info = env.info
    for t in range(task.max_steps):
        act = agent.act(obs, info)
        if len(obs) > 0 and act['primitive']:
            episode.append((obs, act, info))
        (obs, reward, done, info) = env.step(act)
        total_reward += reward
        last_stuff = (obs, info)
        if done:
            break
    return total_reward, t, episode, last_stuff


def is_goal_conditioned(args):
    """
    Be careful with checking this condition. See `load.py`.
    Here, we just check the task name.
    """
    goal_tasks = ['insertion-goal', 'cable-shape-notarget', 'cable-line-notarget',
            'cloth-flat-notarget', 'bag-color-goal']
    return (args.task in goal_tasks)


def ignore_this_demo(args, demo_reward, t, last_extras):
    """In some cases, we should filter out demonstrations.

    Filter for if t == 0, which means the initial state was a success.
    Also, for the bag envs, if we end up in a catastrophic state, I exit
    gracefully and we should avoid those demos (they won't have images we
    need for the dataset anyway).
    """
    ignore = (t == 0)
    if 'exit_gracefully' in last_extras:
        assert last_extras['exit_gracefully']
        return True
    if (args.task in ['bag-color-goal']) and demo_reward <= 0.5:
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disp',      action='store_true')
    parser.add_argument('--task',      default='insertion-goal')
    parser.add_argument('--num_goals', default=20, type=int)
    parser.add_argument('--hz',        default=240.0, type=float)
    args = parser.parse_args()
    assert is_goal_conditioned(args)

    # Initialize environment and task.
    env = Environment(args.disp, hz=args.hz)
    task = tasks.names[args.task]()
    dataset = Dataset(os.path.join('goals', args.task))
    task.mode = 'train'
    seed_to_add = 0

    # For some tasks, call reset() again with a new seed if init state is 'done'.
    while dataset.num_episodes < args.num_goals:
        seed = 10**MAX_ORDER + dataset.num_episodes + seed_to_add
        print(f'\nNEW GOAL: {dataset.num_episodes+1}/{args.num_goals}, seed: {seed}\n')
        np.random.seed(seed)
        demo_reward, t, episode, last_stuff = rollout(task.oracle(env), env, task)
        last_extras = last_stuff[1]['extras']
        if ignore_this_demo(args, demo_reward, t, last_extras):
            seed_to_add += 1
            print(f'Initial state is done() or otherwise we need to ignore, re-sample seed: {seed_to_add}')
        else:
            dataset.add(episode, last_stuff)
            last_extras = last_stuff[1]['extras']
            print(f'\ndemo reward: {demo_reward:0.5f}, len {t}, last_i: {last_extras}')
