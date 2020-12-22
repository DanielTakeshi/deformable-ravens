#!/usr/bin/env python

import os
import sys
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
np.set_printoptions(suppress=True, linewidth=200)
from ravens import utils as U
from ravens import tasks

# List agents by longer name first, for the most precise filtering.
AGENTS = ['-transporter-goal-snaive',
          '-transporter-goal-naive',
          '-transporter-goal',
          '-transporter']

# Task names as strings, REVERSE-sorted so longer names can be used for filtering.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


def get_max_episode_len(results_name):
    """A somewhat more scalable way to get the max episode lengths."""
    task_agent, _ = results_name

    # Remove the agent name (transporter, transporter-goal, etc.).
    for ag in AGENTS:
        if ag in task_agent:
            task_agent = task_agent.replace(ag, '')

    # Get the actual task, plus max steps (remember, subtract one).
    task = tasks.names[task_agent]()
    max_steps = task.max_steps - 1
    return max_steps


def split_task_name_agent(name_task_agent):
    """Split up this name into (task, agent). Makes things easier.

    Note: assumes that `AGENTS` is listed by longer name first, which helps
    with filtering by name. Also assumes that the agent name does not
    coincide with a task name (to a certain extent).

    Parameters
    ----------
    name_task_agent: Look at the STUFF_TO_PLOT dict below. This represents
        the first item in the values, so it's (task)-(agent).
    """
    for ag in AGENTS:
        if ag in name_task_agent:
            task_name = name_task_agent.replace(ag,'')
            assert task_name in TASK_NAMES, f'Error, task: {task_name}?'
            ag = ag[1:]  # Remove leading dash
            return (task_name, ag)
    print(f'Error, {name_task_agent} has no agent in it?')
    sys.exit()


def filter_subdirectories(name_task_agent, name_details):
    """Filters sub-directories for plotting.

    The for loop over `sorted_paths` goes through all the directories within
    `test_results`. We trained with different demos, usually {1,10,100,1000},
    and each demo case will be in its own directory. We combine these demos
    that correspond to this particular task and agent decision, and return
    the resulting directories, along with the number of demos and seeds.

    Parameters
    ----------
    name_task_agent and name_details: Look at the STUFF_TO_PLOT dict below.
        These represent the first and second items in the values. Thus,
        `name_task_agent` consists of (task)-(agent) whereas `name_details`
        has the stuff after the demos/seed count (e.g., indicating number of
        rotations, or anything else we decided to put there).

    Returns
    -------
    (directories, demos, seeds), a tuple where all three items are lists.
    """
    directories = []
    demos = []
    seeds = []
    sorted_paths = sorted(os.listdir(HEAD))

    for dir_name in sorted_paths:
        # This condition tests for everything OTHER than the demo/seed. However
        # it might not be precise if we added stuff to the end of an agent's
        # directory, which hopefully is caught by checking if splitting x by
        # a hyphen leads to a result of length 2.
        if name_task_agent in dir_name and name_details in dir_name:
            # Get demos and seed.
            x = dir_name.replace(name_task_agent, '')
            x = x.replace(name_details, '')
            # x should be '-demos-seed-'
            x = x.lstrip('-').rstrip('-')
            # one last filter just in case
            if len(x.split('-')) != 2:
                continue
            d, s = x.split('-')
            directories.append( join(HEAD,dir_name) )
            demos.append(d)
            seeds.append(s)

    print(f'\n\nPlotting {plot_name} with these directories:')
    for (dire, demo, seed) in zip(directories, demos, seeds):
        print(f'  demos: {str(demo).zfill(4)}, seed {seed}: {dire}')
    return directories, demos, seeds


# -----------------------------------------------------------------------------
# Plot results from the `load.py` script for envs with deformables.
# -----------------------------------------------------------------------------
# stuff_to_plot: each item in this dict will create one figure. Format:
#
#   name: (env+model, other_stuff)
#
# :name: this will go in the actual plot's name.
# :env+model: must be first part of the directory, e.g., cloth-cover-transporter
# :other_stuff: anything after the "demos-seed" to specify the directory name.
# I'm including this to leave flexibility in case we change the later parts of
# the directory names. Note that when making these plots, it is not necessary
# for us to have results from all iterations, e.g., if doing the usual 1K
# through 20K evaluations, we will still generate correct plots with a subset
# of these because we save the iter and use those for the x-axis.
# -----------------------------------------------------------------------------
# (14 Aug 2020) Let's simplify this by merging the different demos together.
# For an env and algorithm we want all the demos {1,10,100,100} on one plot.
# -----------------------------------------------------------------------------
# (24 Aug 2020) Including goal-conditioning envs here, such as insertion-goal,
# even though it's not a 'deformable' ...
# -----------------------------------------------------------------------------
# (26 Aug 2020) Some changes to make goal-conditioning work better. In general,
# we need to be careful about how we filter out the `other_stuff`
# -----------------------------------------------------------------------------
# (02 Sept 2020) Better name filtering, support cable-shape-notarget.
# Deprecated stuff from before Aug 14.
# -----------------------------------------------------------------------------

STUFF_TO_PLOT = {
    'CableRing-Tran-CropAfter':             ('cable-ring-transporter',                     'rots-24-crop_bef_q-0-rotsinf-01'),
    'CableRingNoTarget-Tran-CropAfter':     ('cable-ring-notarget-transporter',            'rots-24-crop_bef_q-0-rotsinf-01'),
    'CableShape-Tran-CropAfter':            ('cable-shape-transporter',                    'rots-24-crop_bef_q-0-rotsinf-01'),
    'ClothFlat-Tran-CropAfter':             ('cloth-flat-transporter',                     'rots-24-crop_bef_q-0-rotsinf-01'),
    'ClothCover-Tran-CropAfter':            ('cloth-cover-transporter',                    'rots-24-crop_bef_q-0-rotsinf-01'),
    'BagAloneOpen-Tran-CropAfter':          ('bag-alone-open-transporter',                 'rots-24-crop_bef_q-0-rotsinf-01'),
    'BagItemsEasy-Tran-CropAfter':          ('bag-items-easy-transporter',                 'rots-24-crop_bef_q-0-rotsinf-01'),
    'CableShapeNoTarget-TranGoal-FinG':     ('cable-line-notarget-transporter-goal',       'rots-24-fin_g-rotsinf-01'),
    'CableShapeNoTarget-TranNaive-FinG':    ('cable-line-notarget-transporter-goal-naive', 'rots-24-fin_g-rotsinf-01'),
    'CableLineNoTarget-TranGoal-FinG':      ('cable-line-notarget-transporter-goal',       'rots-24-fin_g-rotsinf-01'),
    'CableLineNoTarget-TranNaive-FinG':     ('cable-line-notarget-transporter-goal-naive', 'rots-24-fin_g-rotsinf-01'),
    'ClothFlatNoTarget-TranGoal-FinG':      ('cloth-flat-notarget-transporter-goal',       'rots-24-fin_g-rotsinf-01'),
    'ClothFlatNoTarget-TranNaive-FinG':     ('cloth-flat-notarget-transporter-goal-naive', 'rots-24-fin_g-rotsinf-01'),
    'InsertionGoal-TranGoal':               ('insertion-goal-transporter-goal',            'rots-24-fin_g-rotsinf-24'),
    'InsertionGoal-TranNaive':              ('insertion-goal-transporter-goal-naive',      'rots-24-fin_g-rotsinf-24'),
}

#-----------------------------------------------------------------------------
# Matplotlib stuff
#-----------------------------------------------------------------------------

DEMO_TO_COLOR = {
    '1': 'red',
    '10': 'blue',
    '100': 'black',
    '1000': 'gold',
}

errorbars = False
title_size = 34
x_size = 28
tick_size = 28
legend_size = 23
eps = 0.1
lw = 3

HEAD = 'test_results'
ylabel = 'Task Success (%)'
xlabel = 'Training Steps'

#-----------------------------------------------------------------------------
# Generate Deformables Plots from Loading Snapshots (from Training)
#-----------------------------------------------------------------------------

for plot_name, results_name in STUFF_TO_PLOT.items():
    episode_len = get_max_episode_len(results_name)

    # Find sub-directories to combine different demo counts for the same task/agent.
    name_task_agent, name_details = results_name
    name_task, name_agent = split_task_name_agent(name_task_agent)
    directories, demos, seeds = filter_subdirectories(name_task_agent, name_details)
    print(f'On task: {name_task} and agent: {name_agent}')

    # Get plot initialized. 3 cols seems sufficient.
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(10*ncols, 8*nrows))
    title = f'{name_task_agent}_{name_details}'
    supertitle = fig.suptitle(title, fontsize=title_size+2)

    for dire, demo, seed in zip(directories, demos, seeds):
        pickles = sorted([join(dire,x) for x in os.listdir(dire) if x[-4:] == '.pkl'])
        color = DEMO_TO_COLOR[demo]

        # Shared among all envs. These lists contain one item per snapshot.
        x_ticks = []
        y_reward = []
        y_length = []
        y_success = []

        # Specific to certain envs, replacing 'y_reward' for env-specific stuff.
        y_coverage = []
        y_coverage_imp = []
        y_perc_beads = []
        y_perc_imp = []
        y_ssim_imp = []
        y_item_zone = []

        # Iterate through iterations (e.g., itr=1000, then itr=2000, etc).
        for pkl in pickles:
            # ---------------------------------------------------------------- #
            # Each `data` contains results from test-time rollouts from
            # `load.py`. We must have the following starting 12 Aug 2020:
            #
            #   `data`: [(iter, episode_list, dict_results), ...],
            #
            # where EACH TUPLE is ONE test episode. The `iter`s should match
            # among all items in the list.
            # ---------------------------------------------------------------- #
            # In standard ravens, a 'success' means rew == 1 but we should use
            # `dones` for 'y_success', because in many cases we don't have a
            # 'rew == 1' condition, such as with opening a bag area. The "done"
            # condition is true when a task is done CORRECTLY, AND it correctly
            # covers cases when the agent accomplished the task on the last
            # time step. Exceptions to the rule of "done=True means success":
            #
            # (1) bag-items-easy where "done" can be true if reward > 0 at all,
            # but we want reward > 0.5. We filter those in `main.py` and should
            # do something similar here for test-time rollouts.
            #
            # (2) bag-items-hard, a similar thing applies here, however we have
            # to be careful since we can get reward > 0.5 but with only one of
            # the two items in the zone.
            # ---------------------------------------------------------------- #
            data = pickle.load(open(pkl, 'rb'))
            iters = []
            rewards = []
            lengths = []
            coverage = []
            perc_beads = []
            perc_imp = []
            ssim_imp = []
            cov_imp = []
            item_zone = []
            dones = []

            # Loop over `data` list. Each `item` has info from ONE test-time episode.
            for item in data:
                assert len(item) == 3, len(item)

                # Use `first_info` for `info` from the first time step in this episode.
                # episode_list = [(act_1,info_1), ..., (act_T,info_T)]
                itr, episode_list, last_info = item
                first_info = episode_list[0][1]
                assert 'final_info' in last_info, last_info.keys()
                assert 'extras' in first_info, first_info.keys()

                # ------------------------------------------------------------ #
                # Track task-specific metrics for second subplot. For bag-items
                # tasks, need to also append `dones` -- see comments later.
                # ------------------------------------------------------------ #
                if name_task in ['cloth-flat-easy', 'cloth-flat-notarget']:
                    coverage_start = first_info['extras']['cloth_coverage']
                    coverage_final = last_info['final_info']['cloth_coverage']
                    improvement = coverage_final - coverage_start
                    cov_imp.append(improvement)
                elif name_task in ['cable-shape-notarget', 'cable-line-notarget']:
                    nb_zone = last_info['final_info']['nb_zone']
                    nb_beads = last_info['final_info']['nb_beads']
                    frac = nb_zone / nb_beads
                    perc_beads.append(frac)
                elif name_task in ['cable-ring', 'cable-ring-notarget', 'bag-alone-open']:
                    area_start = first_info['extras']['convex_hull_area']
                    area_final = last_info['final_info']['convex_hull_area']
                    percentage = 100 * (area_final - area_start) / area_start
                    perc_imp.append(percentage)
                elif name_task == 'cloth-cover':
                    pass  # Doing nothing here.
                elif name_task == 'bag-items-easy':
                    in_zone = last_info['final_info']['zone_items_rew']
                    item_zone.append(in_zone)
                elif name_task == 'bag-items-hard':
                    in_zone = last_info['final_info']['zone_items_rew']
                    item_zone.append(in_zone)
                elif 'transporter-goal' in name_agent:
                    ssim_start = first_info['image_metrics']['SSIM']
                    ssim_final = last_info['image_metrics']['SSIM']
                    percentage = 100 * (ssim_final - ssim_start) / np.abs(ssim_start)
                    ssim_imp.append(percentage)

                # ------------------------------------------------------------ #
                # We should really be using `task.done`. TODO(daniel): actually
                # this over-counts successes for the bag-items tasks, since we
                # have 'done' whenever we reach task stage 3 and get any
                # reward, but we should be filtering based on if reward > 0.5.
                # See `main.py` and its `ignore_this_demo()`.
                # ------------------------------------------------------------ #
                done = last_info['final_info']['task.done']
                if name_task not in ['bag-items-easy', 'bag-items-hard']:
                    dones.append(done)
                elif name_task == 'bag-items-easy':
                    episode_total_reward = last_info['final_info']['total_rewards']
                    truly_done = done and episode_total_reward > 0.5
                    dones.append(truly_done)
                elif name_task == 'bag-items-hard':
                    episode_total_reward = last_info['final_info']['total_rewards']
                    truly_done = done and episode_total_reward > 0.5
                    dones.append(truly_done)

                # Pretty sure we want total_reward instead of reward.
                # They are not always the same ... e.g., sometimes due to time out errors.
                # AH, for insertion-goal I don't think we added the 'total_rewards' key.
                if 'total_rewards' not in last_info['final_info'].keys():
                    assert name_task == 'insertion-goal', \
                            f'Something went wrong: look at: {last_info}'
                    rewards.append(last_info['reward'])
                else:
                    rewards.append(last_info['final_info']['total_rewards'])
                lengths.append(last_info['length'])

                # `iters` should contain the same value; we test the std later.
                iters.append(itr)

            x_ticks.append(np.mean(iters))
            assert np.std(iters) == 0, iters

            # Add (statistic, standard_deviation) for stuff we plot.
            y_reward.append( (np.mean(rewards), np.std(rewards)) )
            y_length.append( (np.mean(lengths), np.std(lengths)) )

            # ---------------------------------------------------------------- #
            # Record task-specific metrics such as coverage, % improvement,
            # image metrics, etc. We'll use these for the SECOND subplot.
            # ---------------------------------------------------------------- #
            if name_task in ['cloth-flat-easy', 'cloth-flat-notarget']:
                info = (np.mean(cov_imp), np.std(cov_imp))
                y_coverage_imp.append(info)
            elif name_task in ['cable-shape-notarget', 'cable-line-notarget']:
                info = (np.mean(perc_beads), np.std(perc_beads))
                y_perc_beads.append(info)
            elif name_task in ['cable-ring', 'cable-ring-notarget', 'bag-alone-open']:
                info = (np.mean(perc_imp), np.std(perc_imp))
                y_perc_imp.append(info)
            elif name_task == 'cloth-cover':
                pass  # Not doing anything here.
            elif name_task == 'bag-items-easy':
                info = (np.mean(item_zone), np.std(item_zone))
                y_item_zone.append(info)
            elif name_task == 'bag-items-hard':
                info = (np.mean(item_zone), np.std(item_zone))
                y_item_zone.append(info)
            elif 'transporter-goal' in name_agent:
                info = (np.mean(ssim_imp), np.std(ssim_imp))
                y_ssim_imp.append(info)

            # NOTE: `dones` is uniformly applied, for the THIRD subplot.
            success_rate = np.sum(np.array(dones) == 1) / len(dones)
            y_success.append(success_rate)

        # -------------------------------------------------------------------- #
        # (1 of 3) Episode lengths.
        # -------------------------------------------------------------------- #
        y_len  = [x[0] for x in y_length]
        y_lens = [x[1] for x in y_length]
        ep_label = f'Demos {demo}, max {np.max(y_len):0.1f}'
        if errorbars:
            ax[0,0].errorbar(x_ticks, y_len, y_lens, capsize=4, label=ep_label, color=color)
        else:
            ax[0,0].plot(x_ticks, y_len, lw=lw, label=ep_label, color=color)
        ax[0,0].set_title('Episode Lengths', fontsize=title_size)
        ax[0,0].set_ylim([0-eps,episode_len+eps])

        # -------------------------------------------------------------------- #
        # (2 of 3) Reward, must be within [0,1], and is task-specific. By
        # default, we use `y_reward`, which is applied in cable-shape (for
        # example). Use x[0] and x[1] for mean and standard deviation,
        # respectively. Use `factor` to increase the y-axis scale as needed.
        # -------------------------------------------------------------------- #
        y_rew  = [x[0] for x in y_reward]
        y_rews = [x[1] for x in y_reward]
        label = 'Episode Rewards'
        rew_label = f'Ep. Rewards\nMax: {np.max(y_rew):0.2f}'
        factor = 1

        # Over-write (y_rew, y_rews) if desired.
        if name_task in ['cloth-flat-easy', 'cloth-flat-notarget']:
            y_rew  = [x[0] for x in y_coverage_imp]
            y_rews = [x[1] for x in y_coverage_imp]
            label = 'Coverage Improvement'
            rew_label = f'Cov. Imp., max {np.max(y_rew):0.2f}'
        elif name_task in ['cable-shape-notarget', 'cable-line-notarget']:
            y_rew  = [x[0] for x in y_perc_beads]
            y_rews = [x[1] for x in y_perc_beads]
            label = 'Percent of Beads'
            rew_label = f'Perc. Beads, max {np.max(y_rew):0.2f}'
        elif name_task in ['cable-ring', 'cable-ring-notarget', 'bag-alone-open']:
            factor = 100
            y_rew  = [x[0] for x in y_perc_imp]
            y_rews = [x[1] for x in y_perc_imp]
            label = '% Area Improve'
            rew_label = f'% Improve., max {np.max(y_rew):0.2f}'
        elif name_task == 'cloth-cover':
            # Just use the reward.
            pass
        elif name_task == 'bag-items-easy':
            # Could use item zone or just straight up reward? If passing, use reward.
            pass
        elif name_task == 'bag-items-hard':
            # Could use item zone or just straight up reward? If passing, use reward.
            pass
        elif 'transporter-goal' in name_agent:
            factor = 10
            y_rew  = [x[0] for x in y_ssim_imp]
            y_rews = [x[1] for x in y_ssim_imp]
            label = 'SSIM % Improvement'
            rew_label = f'SSIM imp., max {np.max(y_rew):0.2f}'

        # Actually plot.
        if errorbars:
            ax[0,1].errorbar(x_ticks, y_rew, y_rews, capsize=4, label=rew_label, color=color)
        else:
            ax[0,1].plot(x_ticks, y_rew, lw=lw, label=rew_label, color=color)
        ax[0,1].set_title(label, fontsize=title_size)
        ax[0,1].set_ylim([0*factor-eps, 1*factor+eps])

        # -------------------------------------------------------------------- #
        # (3 of 3) Success rates.
        # -------------------------------------------------------------------- #
        print(f'Success rate: {y_success}')
        suc_label = f'Demos {demo}, max {np.max(y_success):0.2f}'
        ax[0,2].plot(x_ticks, y_success, lw=lw, label=suc_label, color=color)
        ax[0,2].set_title('Success Rate', fontsize=title_size)
        ax[0,2].set_ylim([0-eps,1+eps])

        # Bells and whistles.
        for r in range(nrows):
            for c in range(ncols):
                if r == 0:
                    ax[r,c].set_xlabel('Snapshot Train Iter', fontsize=x_size)
                ax[r,c].tick_params(axis='x', labelsize=tick_size)
                ax[r,c].tick_params(axis='y', labelsize=tick_size)
                ax[r,c].legend(prop={'size': legend_size})

    # Shift subplots down due to supertitle.
    plt.tight_layout()
    supertitle.set_y(0.98)
    fig.subplots_adjust(top=0.82)

    # Save file.
    fname = join(os.getcwd(), 'figures', f'plot_{plot_name}.png')
    plt.savefig(fname)
    print(f'Image saved to: {fname}')