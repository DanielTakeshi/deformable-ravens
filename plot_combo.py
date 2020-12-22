#!/usr/bin/env python
"""
Use `plot_defs.py` for quick plotting of all the tasks with separate plots for each.
This one will be used for the actual paper, and show more concise plotting.

Plot results from the `load.py` script for envs with deformables. Modify the
STUFF_TO_PLOT dictionary below,w here the key:value format is:

    name: (env+model, other_stuff, (x,y))

    name: this will show up in the actual plot.
    env+model: must be first part of the directory, e.g., cloth-cover-transporter
    other_stuff: anything after the "demos-seed" to specify the directory name.
    (x,y): these are coordinates in our plotting grid, starting from 0.

I'm including `other_stuff` this to leave flexibility in case we change the
later parts of the directory names. When making these plots, it is not
necessary for us to have results from all iterations, e.g., if doing the
usual 1K through 20K evaluations, we will still generate correct plots with a
subset of these because we save the iter and use those for the x-axis.

NOTE: also use this to print the values we want to report in a Table.
"""
import os
import sys
import pickle
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from os.path import join
from collections import defaultdict
np.set_printoptions(suppress=True, linewidth=200)
from ravens import utils as U
from ravens import tasks
plt.style.use('seaborn')

# List agents by longer name first, for the most precise filtering.
AGENTS = ['-transporter-goal-snaive',
          '-transporter-goal-naive',
          '-transporter-goal',
          '-transporter',
          '-gt_state_2_step',
          '-gt_state']

# Task names as strings, REVERSE-sorted so longer names can be used for filtering.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]

# REPLACE any so that I get stuff I want TO SHOW IN THE PAPER.
REPLACE_TASK = {'cloth-flat':          'fabric-flat',
                'cloth-flat-notarget': 'fabric-flat-notarget',
                'cloth-cover':         'fabric-cover',
                'insertion-goal':      'block-notarget',
                'bag-items-easy':      'bag-items-1',
                'bag-items-hard':      'bag-items-2',}

GOAL_CONDITIONED_TASKS = ['cable-line-notarget',
                          'cable-shape-notarget',
                          'cloth-flat-notarget',
                          'bag-color-goal',
                          'insertion-goal',]

#-----------------------------------------------------------------------------
# MODIFY THIS! Note that we probably want to enable all possible agents.
# TODO: extend this so we don't require -transporter in the name.
#-----------------------------------------------------------------------------

#TAIL1 = 'rots-24-crop_bef_q-0-rotsinf-01'
#TAIL2 = 'rots-24-crop_bef_q-0-rotsinf-01'  # bag-items-hard
#TAIL3 = 'rots-24-fin_g-rotsinf-01'
#TAIL4 = 'rots-24-fin_g-rotsinf-24'  # insertion-goal (block-notarget)

# (1) name for plot to show in paper, (2) name of task used in code, (3) coordinates.
STUFF_TO_PLOT = [
    ('CableRing',          'cable-ring',            (0,0)),
    ('CableRingNoTarget',  'cable-ring-notarget',   (0,1)),
    ('CableShape',         'cable-shape',           (0,2)),
    ('FabricCover',        'cloth-cover',           (0,3)),
    ('FabricFlat',         'cloth-flat',            (1,0)),
    ('BagAloneOpen',       'bag-alone-open',        (1,1)),
    ('BagItems-1',         'bag-items-easy',        (1,2)),
    ('BagItems-2',         'bag-items-hard',        (1,3)),
    ('CableLineNoTarget',  'cable-line-notarget',   (2,0)),
    ('CableShapeNoTarget', 'cable-shape-notarget',  (2,1)),
    ('FabricFlatNoTarget', 'cloth-flat-notarget',   (2,2)),
    ('BagColorGoal',       'bag-color-goal',        (2,3)),
    #('BlockNoTarget',      'insertion-goal',        (2,3)),
]

#-----------------------------------------------------------------------------
# Matplotlib stuff
#-----------------------------------------------------------------------------

DEMO_TO_COLOR = {
    '1': 'red',
    '10': 'blue',
    '100': 'black',
    '1000': 'gold',
}
AGENT_TO_COLOR = {
    'gt_state': 'blue',
    'gt_state_2_step': 'purple',
    'transporter': 'red',
    'transporter-goal': 'red',
    'transporter-goal-naive': 'gold',
    'transporter-goal-snaive': 'orange',
}
AGENT_TO_LABEL = {
    'gt_state':                 'GT-State',
    'gt_state_2_step':          'GT-State 2Step',
    'transporter':              'Transporter',
    'transporter-goal':         'Tran-Goal-Split',
    'transporter-goal-naive':   'Tran-Goal-Combo-Naive',
    'transporter-goal-snaive':  'Tran-Goal-Stack',
}

title_size = 42
x_size = 38
y_size = 38
tick_size = 35
legend_size = 30
lw = 5
ms = 12
EPS = 0.01  # for axes ranges

# HEAD: assumes all stuff is stored in this sub-directory.
HEAD = 'test_results'
ylabel = 'Task Success (%)'
xlabel = 'Train Steps (Thousands)'

#-----------------------------------------------------------------------------
# Helper methods
#-----------------------------------------------------------------------------

def get_max_episode_len(name_task):
    # Get the actual task, plus max steps (remember, subtract one).
    task = tasks.names[name_task]()
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


def get_subdirectories_demos(name_task, num_demos):
    """Filters sub-directories for plotting based on task name and demo count.

    We need to be a little careful with naming conventions due to (a) some tasks
    being supersets of each other, and (b) having different agents and demo counts
    for each task.

    Args:
        name_task: This is the actual name that we used in code. Use name_plot (not
        an argument here) for what we want to show in the plots.
        num_demos: Obvious :-)

    Returns:
        (directories, demos, seeds), a tuple where all three items are lists.
    """
    directories = []
    seeds = []
    sorted_paths = sorted(os.listdir(HEAD))

    # Add other strings here in case we experiment with other suffixes.
    remove_list = [
        '-rots-24-crop_bef_q-0-rotsinf-01',  # filters most of transporter policies
        '-rots-24-crop_bef_q-0-rotsinf-24',  # filters bag-items-hard
        '-rots-24-fin_g-rotsinf-01',         # filters goal-based, other than insertion-goal
        '-rots-1-fin_g-rotsinf-01',          # oops I did one training with rots=1 during training
        '-rots-24-fin_g-rotsinf-24',         # filters goal-based, insertion-goal
        '-rotsinf-01',                       # filters gt_state agents
        '-rotsinf-24',                       # filters gt_state agents for insertion-goal and bag-items-hard
    ]

    for dir_name in sorted_paths:
        # At minimum need to have this condition.
        if name_task not in dir_name:
            continue
        name = dir_name

        # But still have to do extra checks after this. Strip away final suffixes.
        for string in remove_list:
            if string in name:
                name = name.replace(string, '')

        # At this point, we should have:
        #   name = (some task name)-(some agent name)-(demos)-(seed)
        # But to handle overlapping names (e.g., cable-ring vs cable-ring-notarget),
        # we'll attempt to remove the task name, and later remove demos/seed.
        name = name.replace(name_task, '')

        # Now, ideally it is (some agent name)-(demos)-(seed). Let's split:
        x = name.split('-')
        seed = int(x[-1])
        demos = int(x[-2])

        # Check the number of demonstrations.
        if demos != num_demos:
            continue

        # Dump demos and seed. So now we should have (some agent name).
        # But we could also have left behind stuff (e.g., -notarget) so check.
        x = '-'.join(x[:-2])  # Dump demos/seed, re-combine.
        if x not in AGENTS:
            continue

        # Finally!
        directories.append( join(HEAD, dir_name) )
        seeds.append(seed)

    print(f'\nWill plot with these directories:')
    for (dire, seed) in zip(directories, seeds):
        print(f'  for {num_demos}, seed {seed}: {dire}')
    return directories, seeds


def _get_episode_results(args, data, name_task, pkl_file):
    """Given the pickle file which has results (`data`) determine what to plot.

    We could use total_rewards but better to use `task.done`, because total_rewards
    shows the overall sum of delta rewards, but for convex hull tasks, that is not
    interpretable, for cloth and bag-items tasks, we'd rather use coverage or our
    defined success criterion, for cables we probably want percentage of beads in a
    zone, and also total_rewards won't take into account how some initial states
    may start with some reward (especially for cable/cloth stuff), etc. For some
    bag tasks we have to do a little extra filtering. Research code. :-(

    So, to be clear, we use task.done as our metric, EXCEPT for: (a) cable tasks
    (not cable-ring) where we report percentage of beads in a zone, and (b) cloth
    flat stuff, where we report coverage. Otherwise, use `task.done`.

    (13 Oct 2020) realized there is a slight issue, len(episode_list) CAN be 0
    for gt_state agents because before today, I was not saving the initial states.
    AH! This only matters if we wanted to report the gain in reward over the
    starting state. The final state still is OK, and the length is OK since we
    can take the value in the last_info and that is correct. It's just that before
    this, the length of the `episode_list` is one minus that.

    Args:
        data: A list of length 20, since we did 20 test-time episodes per snapshot.
            Each item will contain information about the episode.
        name_task: The task name.
        pkl_file: Pickle file that we are loading from now, useful in case something
            wrong happens.

    Returns:
        Values to report for plotting. The xs is just a scalar representing the
        iteration, which should be the same for all stuff in this `data` file.
        The ys must be whatever we designate for showing results, whether it is
        success rate, coverage, percentage of beads in zone, etc. This is where
        task specific logic occurs.
    """
    total_rewards = []
    lengths = []
    dones = []

    # These are what we actually report! Others above are helpful for debugging.
    iters = []
    metrics = []

    for item in data:
        assert len(item) == 3, len(item)

        # TODO(daniel): argh, see hack above.
        # Use `first_info` for `info` from the first time step in this episode.
        # episode_list = [(act_1,info_1), ..., (act_T,info_T)]
        itr, episode_list, last_info = item
        if len(episode_list) == 0:
            print(f'Note, zero length episode list in {pkl_file}')
            assert 'gt_state' in pkl_file, \
                'We expect this to be a gt_state agent on or before October 13.'
        elif 'gt_state' in pkl_file:
            # AH, if we did gt_state then we don't have 'extras' in the first info.
            # The reason is that first info is directly the env.info, normally we
            # don't add this for image-based policies (since we skip the first
            # addition to the episode_list) but with gt_state we do add it.
            pass
        else:
            # Please fix this!
            first_info = episode_list[0][1]
            assert 'extras' in first_info, \
                f'Issue with {first_info.keys()}\nfor {pkl_file}, length: {len(episode_list)}'
        assert 'final_info' in last_info, last_info.keys()

        # Note: we don't need to report reward or length, leaving here just in case.
        if 'total_rewards' not in last_info['final_info'].keys():
            assert name_task == 'insertion-goal', \
                f'Something went wrong: look at: {last_info} in {pkl_file}'
            total_rewards.append(last_info['reward'])
        else:
            total_rewards.append(last_info['final_info']['total_rewards'])
        lengths.append(last_info['length'])
        iters.append(itr)

        # ------------------------------------------------------------------------------ #
        # We should really be using `task.done`. Actually this over-counts successes for
        # bag-items tasks, since we have 'done' whenever we reach task stage 3 and get any
        # reward, but we should be filtering based on if reward > 0.5. See `main.py` and
        # its `ignore_this_demo()`.
        # Note: we can try putting fraction of item into target for bag-color-goal?
        # ------------------------------------------------------------------------------ #
        done = last_info['final_info']['task.done']
        if name_task not in ['bag-items-easy', 'bag-items-hard']:
            dones.append(done)
        elif name_task == 'bag-items-easy':
            episode_total_reward = last_info['final_info']['total_rewards']
            truly_done = done and episode_total_reward > 0.5
            dones.append(truly_done)
        elif name_task == 'bag-items-hard':
            truly_done = (done and
                         (last_info['final_info']['zone_items_rew'] == 0.5) and
                         (last_info['final_info']['zone_beads_rew'] > 0))
            dones.append(truly_done)

        # Now for the actual metric. Put this in a list (not np.array). Don't forget to
        # adjust the plot labels if necessary (outside this code).
        if args.binary:
            metrics.append(dones[-1])
        else:
            if name_task in ['cloth-flat', 'cloth-flat-notarget']:
                coverage = last_info['final_info']['cloth_coverage']
                metrics.append(coverage)
            elif name_task in ['cable-shape', 'cable-shape-notarget', 'cable-line-notarget']:
                nb_zone = last_info['final_info']['nb_zone']
                nb_beads = last_info['final_info']['nb_beads']
                frac = nb_zone / nb_beads
                metrics.append(frac)
            elif name_task in ['bag-color-goal']:
                # I think this is fair, gives partial credit to stuff inside the bag.
                frac = last_info['final_info']['frac_in_target_bag']
                metrics.append(frac)
            else:
                metrics.append(dones[-1])

    # The iteration should be the same for all in `data`.
    assert np.std(iters) == 0, iters
    xs = int(np.mean(iters))

    # Note: do not make `ys` an np.array since we use `extend` later.
    ys = metrics
    return {'x': xs, 'y': ys, 'total_rewards': total_rewards, 'lengths': lengths, 'dones': dones}


def _get_results(name_task, directories, seeds, debug):
    """Store results here to map agent_type --> performance, for x and y values.

    The x's tested should be the same: {2K,4K,6K,8K,10K,12K,14K,16K,18K,20K}.
    To be clear, stats[agent] = [(itr_1, results_1), ..., (itr_N, results_N)],
    where each itr_k is an integer iteration and each results_k is a LIST.
    Then later, we combine all results_k lists for the same itr. That gives us
    60 values (20 episodes x 3 random seeds) to report for each itr.

    The reuslts come from another helper, _get_episode_results(), that processes
    pickle files and determines what to return.

    Returns:
        stats_combo: defaultdict. This is like stats, except we combine different
            random seeds together (we normally ran 3 seeds of training), so that
            we map from a single iteration to all its relevant statistics. It's
            a DICT OF DICTS. The keys in `stats_combo` are agents. The keys in
            `stats_combo[agent]` are iterations.
    """
    stats = defaultdict(list)

    # Each directory within `directories` has a list of pickle files.
    for dire, seed in zip(directories, seeds):
        pickles = sorted([join(dire,x) for x in os.listdir(dire) if x[-4:] == '.pkl'])

        # Each pickle file within `pickles` will have a list of test-time results.
        # This loop effectively loops over iterations: 2K, 4K, etc.
        for pkl in pickles:
            data = pickle.load(open(pkl, 'rb'))
            if len(data) < 20:
                print(f'Warning, len(data): {len(data)}, we did 20 test-time episodes')
            results = _get_episode_results(args, data, name_task, pkl)

            # We've gathered statistics, but now need to assign to an agent type.
            # CAREFUL! Due to overlapping names, test LONGEST names first!
            added = False
            for agent in AGENTS:
                if agent in dire:
                    stats[agent].append( (results['x'], results['y']) )
                    added = True
                    break
            if not added:
                print(f'Error, have not added agent for: {dire}')

    # Combine different iters, e.g., [10K, res_1], [10K, res2] --> [10K, res1, res2].
    stats_combo = {}
    for agent in sorted(stats.keys()):
        itrs_combo = {}
        for item in stats[agent]:
            itr, res = item
            if itr in itrs_combo:
                itrs_combo[itr].extend(res)
            else:
                itrs_combo[itr] = res
        stats_combo[agent] = itrs_combo

    # Inspect the key
    if debug:
        for agent in sorted(stats_combo.keys()):
            print(f'stats_combo[{agent}]:')
            for itr in sorted(stats_combo[agent].keys()):
                res_l = stats_combo[agent][itr]
                print(f'\t{str(itr).zfill(5)}, results (len {len(res_l)}) mean/std (len): '
                        f'{np.mean(res_l):-1.3f} +/- {np.std(res_l):0.1f}')

    return stats_combo


def _check_xvals(xvals):
    ideal = [ 0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]
    for i in ideal:
        if i not in xvals:
            print(f'Warning, {i} not in: {xvals}')

#-----------------------------------------------------------------------------
# Plotting methods.
#-----------------------------------------------------------------------------

def plot_combo_fixed_demos(args, num_demos, standard_error=False, debug=False):
    """Plot training curves of everything together, conditioned on num demos.

    We want to create a grid of all the tasks, and where we have different
    methods within each grid.

    Each `data` (i.e., pickle file) contains results from test-time rollouts
    from `load.py` and consists of the following:

        `data`: [(iter, episode_list, dict_results), ...],

    where EACH TUPLE is ONE test episode. The `iter`s should match among all
    items in a single list.

    In standard ravens, a 'success' means rew == 1 but we should use `dones`
    for 'y_success', because in many cases we don't have a 'rew == 1'
    condition, such as with opening a bag area. The "done" condition is true
    when a task is done CORRECTLY, AND it correctly covers cases when the
    agent accomplished the task on the last time step. Exceptions to the rule
    of "done=True means success":

    (1) bag-items-easy where "done" can be true if reward > 0 at all, but we
    want reward > 0.5. We filter those in `main.py` and should do something
    similar here for test-time rollouts.

    (2) bag-items-hard, a similar thing applies, however be careful since we
    can get reward > 0.5 but with only one of the two items in the zone.
    """
    nrows, ncols = 3, 4
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(8.5*ncols, 9.5*nrows))
    title = f'Training Results, {num_demos} Demos'
    supertitle = fig.suptitle(title, fontsize=title_size+2)

    for item in STUFF_TO_PLOT:
        name_plot, name_task, rc = item
        row, column = rc

        # Extract all relevant subdirectories, and get results.
        directories, seeds = get_subdirectories_demos(name_task, num_demos)
        stats_combo = _get_results(name_task, directories, seeds, debug)

        # For each agent, plot. Iterate through keys of stats_combo[agent] IN ORDER.
        for agent in sorted(stats_combo.keys()):
            results = stats_combo[agent]
            x_vals = []
            y_vals = []
            y_errs = []  # standard deviations
            y_stde = []  # standard errors of mean
            for itr in sorted(stats_combo[agent].keys()):
                x_vals.append( itr )
                values = stats_combo[agent][itr]
                if (len(values) != 60 and name_task != 'bag-color-goal' and itr != 0):
                    print(f'\tWarning, len(values): {len(values)} for: {agent}, {itr}')
                elif (len(values) != 20 and name_task == 'bag-color-goal'):
                    print(f'\tWarning, len(values): {len(values)} for: {agent}, {itr}')
                y_vals.append( np.mean(values) )
                y_errs.append( np.std(values) )
                y_stde.append( np.std(values) / np.sqrt(len(values)) )
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)
            y_errs = np.array(y_errs)
            y_stde = np.array(y_stde)

            # NOTE: we're dividing by thousands here.
            x_vals = x_vals / 1000
            _check_xvals(x_vals)

            # Remove leading dash, and then plot, while clipping stdev's to range.
            ag = agent[1:]
            agent_label = AGENT_TO_LABEL[ag]
            color = AGENT_TO_COLOR[ag]
            if standard_error:
                lower = np.clip(y_vals - y_stde, 0.0 - EPS, 1.0 + EPS)
                upper = np.clip(y_vals + y_stde, 0.0 - EPS, 1.0 + EPS)
            else:
                lower = np.clip(y_vals - y_errs, 0.0 - EPS, 1.0 + EPS)
                upper = np.clip(y_vals + y_errs, 0.0 - EPS, 1.0 + EPS)
            ax[rc].plot(x_vals, y_vals, lw=lw, color=color, ms=ms, marker='o', label=agent_label)
            ax[rc].fill_between(x_vals, upper, lower, color=color, alpha=0.3, linewidth=0.0)
            ax[rc].legend(prop={'size': legend_size})

        # Bells and whistles.
        subplot_title = f'{name_plot}'
        ax[rc].set_title(subplot_title, fontsize=title_size)
        if row == nrows - 1:
            ax[rc].set_xlabel(xlabel, fontsize=x_size)
        ylabel = 'Success Rate'
        if not args.binary:
            if name_task in ['cloth-flat', 'cloth-flat-notarget']:
                ylabel = 'Zone Coverage'
            elif name_task in ['cable-shape', 'cable-shape-notarget', 'cable-line-notarget']:
                ylabel = 'Percent in Zone'
        ax[rc].set_ylabel(ylabel, fontsize=y_size)
        ax[rc].set_ylim([0.0 - EPS, 1.0 + EPS])
        ax[rc].tick_params(axis='x', labelsize=tick_size)
        ax[rc].tick_params(axis='y', labelsize=tick_size)

    # Shift subplots down due to supertitle.
    plt.tight_layout()
    supertitle.set_y(0.98)
    fig.subplots_adjust(top=0.94)

    # Save file.
    suffix = f'plot_demos_{str(num_demos).zfill(4)}_binary_{args.binary}_v03.png'
    fname = join(os.getcwd(), 'figures', suffix)
    plt.savefig(fname)
    print(f'\nImage saved to: {fname}')


def plot_single(args, goal_conditioned, name_task, name_plot):
    """Plot only one thing, similar to the table method.

    Let's not plot transporter-goal-snaive.
    """
    IGNORE = ['-transporter-goal-naive']

    # Override any parameters here.
    title_size = 40
    x_size = 38
    y_size = 38
    tick_size = 35
    legend_size = 25
    lw = 5
    ms = 12

    # Now make the plot.
    nrows, ncols = 1, 4
    fig, ax = plt.subplots(nrows, ncols, squeeze=True, figsize=(8.0*ncols, 9.0*nrows))
    title = f'Training Results, {name_plot}'
    supertitle = fig.suptitle(title, fontsize=title_size+2)

    for col, num_demos in enumerate([1, 10, 100, 1000]):
        # Extract all relevant subdirectories, and get results.
        directories, seeds = get_subdirectories_demos(name_task, num_demos)
        stats_combo = _get_results(name_task, directories, seeds, debug)

        # For each agent, plot. Iterate through keys of stats_combo[agent] IN ORDER.
        for agent in sorted(stats_combo.keys()):
            if agent in IGNORE:
                print(f'  ignoring {agent} for demos {num_demos}')
                continue
            x_vals = []
            y_vals = []
            y_errs = []  # standard deviations
            y_stde = []  # standard errors of mean
            print(f'on agent {agent} with keys {stats_combo[agent].keys()}')
            for itr in sorted(stats_combo[agent].keys()):
                x_vals.append( itr )
                values = stats_combo[agent][itr]
                if (len(values) != 60 and name_task != 'bag-color-goal' and itr != 0):
                    print(f'\tWarning, len(values): {len(values)} for: {agent}, {itr}')
                elif (len(values) != 20 and name_task == 'bag-color-goal'):
                    print(f'\tWarning, len(values): {len(values)} for: {agent}, {itr}')
                y_vals.append( np.mean(values) )
                y_errs.append( np.std(values) )
                y_stde.append( np.std(values) / np.sqrt(len(values)) )
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)
            y_errs = np.array(y_errs)
            y_stde = np.array(y_stde)

            # NOTE: we're dividing by thousands here.
            x_vals = x_vals / 1000

            # Remove leading dash, and then plot, while clipping stdev's to range.
            ag = agent[1:]
            agent_label = AGENT_TO_LABEL[ag]
            color = AGENT_TO_COLOR[ag]
            if standard_error:
                lower = np.clip(y_vals - y_stde, 0.0 - EPS, 1.0 + EPS)
                upper = np.clip(y_vals + y_stde, 0.0 - EPS, 1.0 + EPS)
            else:
                lower = np.clip(y_vals - y_errs, 0.0 - EPS, 1.0 + EPS)
                upper = np.clip(y_vals + y_errs, 0.0 - EPS, 1.0 + EPS)
            ax[col].plot(x_vals, y_vals, lw=lw, color=color, ms=ms, marker='o', label=agent_label)
            ax[col].fill_between(x_vals, upper, lower, color=color, alpha=0.3, linewidth=0.0)
            ax[col].legend(prop={'size': legend_size})

        # Bells and whistles. For y-axis label, just put it on the leftmost column.
        subplot_title = f'Train Demos: {num_demos}'
        ax[col].set_title(subplot_title, fontsize=title_size)
        ax[col].set_xlabel(xlabel, fontsize=x_size)
        ylabel = 'Success Rate'
        if not args.binary:
            if name_task in ['cloth-flat', 'cloth-flat-notarget']:
                ylabel = 'Zone Coverage'
            elif name_task in ['cable-shape', 'cable-shape-notarget', 'cable-line-notarget']:
                ylabel = 'Percent in Zone'
        if col == 0:
            ax[col].set_ylabel(ylabel, fontsize=y_size)
        ax[col].set_ylim([0.0 - EPS, 1.0 + EPS])
        ax[col].tick_params(axis='x', labelsize=tick_size)
        ax[col].tick_params(axis='y', labelsize=tick_size)

    # Shift subplots down due to supertitle.
    plt.tight_layout()
    supertitle.set_y(0.98)
    fig.subplots_adjust(top=0.85)

    # Save file.
    suffix = f'plot_demos_task_{name_task}_v02.png'
    fname = join(os.getcwd(), 'figures', suffix)
    plt.savefig(fname)
    print(f'\nImage saved to: {fname}')

#-----------------------------------------------------------------------------
# Table methods (i.e., these will print values we can copy/paste into a table)
#-----------------------------------------------------------------------------

def print_table(args):
    """Use this for the broad overview table in the paper, showing convergence results.

    Careful: we use some braces, so could be tricky to integrate with string formatting?
    Remember that for line breaks we need an escape character for \, so \\.
    NOTE: Put this between \toprule and \bottomrule commands in LaTeX for tables.
    """
    s = ''

    # Let's just use a fixed orering. Adjust here if needed. This is the task name IN CODE.
    tasks_l = ['cable-ring', 'cable-ring-notarget', 'cable-shape', 'cloth-cover',
               'cloth-flat', 'bag-alone-open', 'bag-items-easy', 'bag-items-hard',
               'cable-line-notarget', 'cable-shape-notarget', 'cloth-flat-notarget', 'bag-color-goal',]

    # This has the same ordering as above but with fixed names (e.g., cloth --> fabric).
    T = []
    for t in tasks_l:
        if t in REPLACE_TASK:
            T.append(REPLACE_TASK[t])
        else:
            T.append(t)

    # Map from (name_task, num_demos) --> statistic.
    a1 = defaultdict(list)
    a2 = defaultdict(list)
    a3 = defaultdict(list)
    a4 = defaultdict(list)
    a5 = defaultdict(list)
    a6 = defaultdict(list)
    a7 = defaultdict(list)

    # For each task, get relevant directories for each demo, then relevant stats.
    for name_task in tasks_l:
        for num_demos in [1, 10, 100, 1000]:
            directories, seeds = get_subdirectories_demos(name_task, num_demos)
            stats_combo = _get_results(name_task, directories, seeds, debug)
            agents = sorted(stats_combo.keys())
            print(f'For task {name_task}, we have results from agents: {agents}')

            # For now, take the max over the iterations (easy to spot check). Take maximum.
            def get_max(_name_task, ag_key):
                stat_max = -1
                for key in sorted(stats_combo[ag_key].keys()):
                    if (len(stats_combo[ag_key][key]) != 60 and name_task != 'bag-color-goal' and key != 0):
                        print(f'\tWarning, len(stats_combo[ag_key][key]): {len(stats_combo[ag_key][key])} for: {ag_key}, {key}')
                    elif len(stats_combo[ag_key][key]) != 20 and name_task == 'bag-color-goal':
                        print(f'\tWarning, len(stats_combo[ag_key][key]): {len(stats_combo[ag_key][key])} for: {ag_key}, {key}')
                    stat_max = max(stat_max, np.mean(stats_combo[ag_key][key]))
                stat_max *= 100
                if _name_task in REPLACE_TASK:
                    return (REPLACE_TASK[_name_task], stat_max)
                else:
                    return (_name_task, stat_max)

            if '-transporter' in agents:
                i1, i2 = get_max(_name_task=name_task, ag_key='-transporter')
                a1[i1].append(i2)
            if '-transporter-goal' in agents:
                i1, i2 = get_max(_name_task=name_task, ag_key='-transporter-goal')
                a2[i1].append(i2)
            if '-transporter-goal-naive' in agents:
                i1, i2 = get_max(_name_task=name_task, ag_key='-transporter-goal-naive')
                a3[i1].append(i2)
            if '-gt_state' in agents:
                i1, i2 = get_max(_name_task=name_task, ag_key='-gt_state')
                a4[i1].append(i2)
            if '-gt_state_2_step' in agents:
                i1, i2 = get_max(_name_task=name_task, ag_key='-gt_state_2_step')
                a5[i1].append(i2)
            if '-conv_mlp' in agents:
                i1, i2 = get_max(_name_task=name_task, ag_key='-gt_state_2_step')
                a6[i1].append(i2)
            if '-transporter-goal-snaive' in agents:
                i1, i2 = get_max(_name_task=name_task, ag_key='-transporter-goal-snaive')
                a7[i1].append(i2)

    print('\nDebugging of which results we have:')
    print(f'keys in a1 (transporter):              {sorted(a1.keys())}')
    print(f'keys in a2 (transporter-goal):         {sorted(a2.keys())}')
    print(f'keys in a3 (transporter-goal-naive):   {sorted(a3.keys())}')
    print(f'keys in a4 (gt_state):                 {sorted(a4.keys())}')
    print(f'keys in a5 (gt_state_2_step):          {sorted(a5.keys())}')
    print(f'keys in a6 (conv_mlp):                 {sorted(a6.keys())}')
    print(f'keys in a7 (transporter-goal-S-naive): {sorted(a7.keys())}')
    NA = 'N/A'

    # Use a1 through a5 default dicts. Keys are tasks in `T` list.
    # For any valid task `i`, we have `a1[T[i]]` a list of length 4, for the 4 demo types.
    # Therefore the LAST index should range from: [0], [1], [2], to [3].
    # If any lists are not of length 4, we should not use them, or simply fill in a value of -1?
    # This will be invalid for some but hopefully those can be easily inspected.

    s += f'\n & \multicolumn{{4}}{{c}}{{{T[0]}}} & \multicolumn{{4}}{{c}}{{{T[1]}}} & \multicolumn{{4}}{{c}}{{{T[2]}}} & \multicolumn{{4}}{{c}}{{{T[3]}}}  \\\\'
    s += '\n \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}'
    s += '\n Method & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000 \\\\'
    s += '\n \midrule'
    s += (f'\n GT-State MLP            & {a4[ T[0] ][0]:0.1f} & {a4[ T[0] ][1]:0.1f} & {a4[ T[0] ][2]:0.1f} & {a4[ T[0] ][3]:0.1f} '  # NOTE! Everything in this row must be 'a4', etc.
                                     f'& {a4[ T[1] ][0]:0.1f} & {a4[ T[1] ][1]:0.1f} & {a4[ T[1] ][2]:0.1f} & {a4[ T[1] ][3]:0.1f} '
                                     f'& {a4[ T[2] ][0]:0.1f} & {a4[ T[2] ][1]:0.1f} & {a4[ T[2] ][2]:0.1f} & {a4[ T[2] ][3]:0.1f} '
                                     f'& {a4[ T[3] ][0]:0.1f} & {a4[ T[3] ][1]:0.1f} & {a4[ T[3] ][2]:0.1f} & {a4[ T[3] ][3]:0.1f}  \\\\')

    s += (f'\n GT-State MLP 2-Step     & {a5[ T[0] ][0]:0.1f} & {a5[ T[0] ][1]:0.1f} & {a5[ T[0] ][2]:0.1f} & {a5[ T[0] ][3]:0.1f} '
                                     f'& {a5[ T[1] ][0]:0.1f} & {a5[ T[1] ][1]:0.1f} & {a5[ T[1] ][2]:0.1f} & {a5[ T[1] ][3]:0.1f} '
                                     f'& {a5[ T[2] ][0]:0.1f} & {a5[ T[2] ][1]:0.1f} & {a5[ T[2] ][2]:0.1f} & {a5[ T[2] ][3]:0.1f} '
                                     f'& {a5[ T[3] ][0]:0.1f} & {a5[ T[3] ][1]:0.1f} & {a5[ T[3] ][2]:0.1f} & {a5[ T[3] ][3]:0.1f}  \\\\')

    s += (f'\n Transporter             & {a1[ T[0] ][0]:0.1f} & {a1[ T[0] ][1]:0.1f} & {a1[ T[0] ][2]:0.1f} & {a1[ T[0] ][3]:0.1f} '  # NOTE! Everything in this row must be 'a1', etc.
                                     f'& {a1[ T[1] ][0]:0.1f} & {a1[ T[1] ][1]:0.1f} & {a1[ T[1] ][2]:0.1f} & {a1[ T[1] ][3]:0.1f} '
                                     f'& {a1[ T[2] ][0]:0.1f} & {a1[ T[2] ][1]:0.1f} & {a1[ T[2] ][2]:0.1f} & {a1[ T[2] ][3]:0.1f} '
                                     f'& {a1[ T[3] ][0]:0.1f} & {a1[ T[3] ][1]:0.1f} & {a1[ T[3] ][2]:0.1f} & {a1[ T[3] ][3]:0.1f}  \\\\')
    s += '\n \midrule'
    s += f'\n & \multicolumn{{4}}{{c}}{{{T[4]}}} & \multicolumn{{4}}{{c}}{{{T[5]}}} & \multicolumn{{4}}{{c}}{{{T[6]}}} & \multicolumn{{4}}{{c}}{{{T[7]}}}  \\\\'
    s += '\n \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}'
    s += '\n Method & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000 \\\\'
    s += '\n \midrule'
    s += (f'\n GT-State MLP            & {a4[ T[4] ][0]:0.1f} & {a4[ T[4] ][1]:0.1f} & {a4[ T[4] ][2]:0.1f} & {a4[ T[4] ][3]:0.1f} '  # NOTE! Everything in this row must be 'a4', etc.
                                     f'& {a4[ T[5] ][0]:0.1f} & {a4[ T[5] ][1]:0.1f} & {a4[ T[5] ][2]:0.1f} & {a4[ T[5] ][3]:0.1f} '
                                     f'& {a4[ T[6] ][0]:0.1f} & {a4[ T[6] ][1]:0.1f} & {a4[ T[6] ][2]:0.1f} & {a4[ T[6] ][3]:0.1f} '
                                     f'& {a4[ T[7] ][0]:0.1f} & {a4[ T[7] ][1]:0.1f} & {a4[ T[7] ][2]:0.1f} & {a4[ T[7] ][3]:0.1f}  \\\\')

    s += (f'\n GT-State MLP 2-Step     & {a5[ T[4] ][0]:0.1f} & {a5[ T[4] ][1]:0.1f} & {a5[ T[4] ][2]:0.1f} & {a5[ T[4] ][3]:0.1f} '  # NOTE! Everything in this row must be 'a5', etc.
                                     f'& {a5[ T[5] ][0]:0.1f} & {a5[ T[5] ][1]:0.1f} & {a5[ T[5] ][2]:0.1f} & {a5[ T[5] ][3]:0.1f} '
                                     f'& {a5[ T[6] ][0]:0.1f} & {a5[ T[6] ][1]:0.1f} & {a5[ T[6] ][2]:0.1f} & {a5[ T[6] ][3]:0.1f} '
                                     f'& {a5[ T[7] ][0]:0.1f} & {a5[ T[7] ][1]:0.1f} & {a5[ T[7] ][2]:0.1f} & {a5[ T[7] ][3]:0.1f}  \\\\')

    s += (f'\n Transporter             & {a1[ T[4] ][0]:0.1f} & {a1[ T[4] ][1]:0.1f} & {a1[ T[4] ][2]:0.1f} & {a1[ T[4] ][3]:0.1f} '  # NOTE! Everything in this row must be 'a1', etc.
                                     f'& {a1[ T[5] ][0]:0.1f} & {a1[ T[5] ][1]:0.1f} & {a1[ T[5] ][2]:0.1f} & {a1[ T[5] ][3]:0.1f} '
                                     f'& {a1[ T[6] ][0]:0.1f} & {a1[ T[6] ][1]:0.1f} & {a1[ T[6] ][2]:0.1f} & {a1[ T[6] ][3]:0.1f} '
                                     f'& {a1[ T[7] ][0]:0.1f} & {a1[ T[7] ][1]:0.1f} & {a1[ T[7] ][2]:0.1f} & {a1[ T[7] ][3]:0.1f}  \\\\')
    s += '\n \midrule'
    s += f'\n & \multicolumn{{4}}{{c}}{{{T[8]}}} & \multicolumn{{4}}{{c}}{{{T[9]}}} & \multicolumn{{4}}{{c}}{{{T[10]}}} & \multicolumn{{4}}{{c}}{{{T[11]}}}  \\\\'
    s += '\n \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}'
    s += '\n Method & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000 & 1 & 10 & 100 & 1000  \\\\'
    s += '\n \midrule'
    s += (f'\n GT-State MLP            & {a4[ T[8] ][0]:0.1f} & {a4[ T[8] ][1]:0.1f} & {a4[ T[8] ][2]:0.1f} & {a4[ T[8] ][3]:0.1f} '
                                     f'& {a4[ T[9] ][0]:0.1f} & {a4[ T[9] ][1]:0.1f} & {a4[ T[9] ][2]:0.1f} & {a4[ T[9] ][3]:0.1f} '
                                     f'& {a4[ T[10]][0]:0.1f} & {a4[ T[10]][1]:0.1f} & {a4[ T[10]][2]:0.1f} & {a4[ T[10]][3]:0.1f} '
                                     f'& {a4[ T[11]][0]:0.1f} & {a4[ T[11]][1]:0.1f} & {a4[ T[11]][2]:0.1f} & {a4[ T[11]][3]:0.1f}  \\\\')

    s += (f'\n GT-State MLP 2-Step     & {a5[ T[8] ][0]:0.1f} & {a5[ T[8] ][1]:0.1f} & {a5[ T[8] ][2]:0.1f} & {a5[ T[8] ][3]:0.1f} '
                                     f'& {a5[ T[9] ][0]:0.1f} & {a5[ T[9] ][1]:0.1f} & {a5[ T[9] ][2]:0.1f} & {a5[ T[9] ][3]:0.1f} '
                                     f'& {a5[ T[10]][0]:0.1f} & {a5[ T[10]][1]:0.1f} & {a5[ T[10]][2]:0.1f} & {a5[ T[10]][3]:0.1f} '
                                     f'& {a5[ T[11]][0]:0.1f} & {a5[ T[11]][1]:0.1f} & {a5[ T[11]][2]:0.1f} & {a5[ T[11]][3]:0.1f}  \\\\')

    s += (f'\n Transporter-Goal-Stack  & {a7[ T[8] ][0]:0.1f} & {a7[ T[8] ][1]:0.1f} & {a7[ T[8] ][2]:0.1f} & {a7[ T[8] ][3]:0.1f} '  # NOTE! Everything in this row must be 'a3'. (Update: actually a7)
                                     f'& {a7[ T[9] ][0]:0.1f} & {a7[ T[9] ][1]:0.1f} & {a7[ T[9] ][2]:0.1f} & {a7[ T[9] ][3]:0.1f} '
                                     f'& {a7[ T[10]][0]:0.1f} & {a7[ T[10]][1]:0.1f} & {a7[ T[10]][2]:0.1f} & {a7[ T[10]][3]:0.1f} '  # If insertion-goal, then keep last columns at a3
                                     f'& {a7[ T[11]][0]:0.1f} & {a7[ T[11]][1]:0.1f} & {a7[ T[11]][2]:0.1f} & {a7[ T[11]][3]:0.1f}  \\\\')

    s += (f'\n Transporter-Goal-Split  & {a2[ T[8] ][0]:0.1f} & {a2[ T[8] ][1]:0.1f} & {a2[ T[8] ][2]:0.1f} & {a2[ T[8] ][3]:0.1f} '  # NOTE! Everything in this row must be 'a2', etc.
                                     f'& {a2[ T[9] ][0]:0.1f} & {a2[ T[9] ][1]:0.1f} & {a2[ T[9] ][2]:0.1f} & {a2[ T[9] ][3]:0.1f} '
                                     f'& {a2[ T[10]][0]:0.1f} & {a2[ T[10]][1]:0.1f} & {a2[ T[10]][2]:0.1f} & {a2[ T[10]][3]:0.1f} '
                                     f'& {a2[ T[11]][0]:0.1f} & {a2[ T[11]][1]:0.1f} & {a2[ T[11]][2]:0.1f} & {a2[ T[11]][3]:0.1f}  \\\\')
    print(s)


def print_single(args, goal_conditioned, name_task, name_plot):
    """Use this for printing a SINGLE item, for single inspection.

    name_task: what we used in code. name_plot: what we want to show in the plot.
    For now, take the max over the iterations (easy to spot check w/curves).
    Actually we have two cases to watch out for (goal conditioned or not)...

    For the table, I'm going to add a few more commands to make it a little easier to copy/paste.
    """
    def get_max(stats_combo, ag_key):
        stat_max = -1
        for key in sorted(stats_combo[ag_key].keys()):
            stat_max = max(stat_max, np.mean(stats_combo[ag_key][key]))
        stat_max *= 100  # We want this in %, in [0,100]
        return stat_max

    # Map from num_demos ---> statistic. Note: DIFFERENT from earlier where we had lists.
    # To be honest this is more scalable as we don't need to wait for all demos to finish.
    # Earlier we appended w/demos, and caused issues if we were missing early demo counts.
    a1, a2, a3, a4, a5, a6, a7 = {}, {}, {}, {}, {}, {}, {}

    # Get relevant directories for each demo, then relevant stats.
    for num_demos in [1, 10, 100, 1000]:
        directories, seeds = get_subdirectories_demos(name_task, num_demos)
        stats_combo = _get_results(name_task, directories, seeds, debug)
        agents = sorted(stats_combo.keys())
        print(f'For task {name_task}, we have results from agents: {agents}')
        if '-transporter' in agents:
            a1[num_demos] = get_max(stats_combo, ag_key='-transporter')
        if '-transporter-goal' in agents:
            a2[num_demos] = get_max(stats_combo, ag_key='-transporter-goal')
        if '-transporter-goal-naive' in agents:
            a3[num_demos] = get_max(stats_combo, ag_key='-transporter-goal-naive')
        if '-gt_state' in agents:
            a4[num_demos] = get_max(stats_combo, ag_key='-gt_state')
        if '-gt_state_2_step' in agents:
            a5[num_demos] = get_max(stats_combo, ag_key='-gt_state_2_step')
        if '-conv_mlp' in agents:
            a6[num_demos] = get_max(stats_combo, ag_key='-gt_state_2_step')
        if '-transporter-goal-snaive' in agents:
            a7[num_demos] = get_max(stats_combo, ag_key='-transporter-goal-snaive')

    print('\nDebugging of which results we have:')
    print(f'keys in a1 (transporter):              {sorted(a1.keys())}')
    print(f'keys in a2 (transporter-goal-split):   {sorted(a2.keys())}')
    print(f'keys in a3 (transporter-goal-naive):   {sorted(a3.keys())} [note should not use this]')
    print(f'keys in a4 (gt_state):                 {sorted(a4.keys())}')
    print(f'keys in a5 (gt_state_2_step):          {sorted(a5.keys())}')
    print(f'keys in a6 (conv_mlp):                 {sorted(a6.keys())}')
    print(f'keys in a7 (transporter-goal-stack): {sorted(a7.keys())}')

    # Manually replace keys with N if we don't have it (I know, I know ...)
    N = ''

    # Use a1 through a5 default dicts. Keys are NUM DEMOS, not tasks.
    s = '\n  \\begin{tabular}{@{}lrrrr}'
    s += '\n  \\toprule'
    s += f'\n  & \multicolumn{{4}}{{c}}{{{name_plot}}} \\\\'
    s += '\n  \cmidrule(lr){2-5}'
    s += '\n  Method & 1 & 10 & 100 & 1000 \\\\'
    s += '\n  \midrule'
    s += (f'\n  GT-State MLP            & {a4[1]:0.1f} & {a4[10]:0.1f} & {a4[100]:0.1f} & {a4[1000]:0.1f} \\\\')
    s += (f'\n  GT-State MLP 2-Step     & {a5[1]:0.1f} & {a5[10]:0.1f} & {a5[100]:0.1f} & {a5[1000]:0.1f} \\\\')
    if goal_conditioned:
        s += (f'\n  Transporter-Goal-Stack  & {a7[1]:0.1f} & {a7[10]:0.1f} & {a7[100]:0.1f} & {a7[1000]:0.1f} \\\\')  # should be a7
        s += (f'\n  Transporter-Goal-Split  & {a2[1]:0.1f} & {a2[10]:0.1f} & {a2[100]:0.1f} & {a2[1000]:0.1f} \\\\')
    else:
        s += (f'\n  Transporter             & {a1[1]:0.1f} & {a1[10]:0.1f} & {a1[100]:0.1f} & {a1[1000]:0.1f} \\\\')
    s += '\n  \\toprule'
    s += '\n  \\end{tabular}'
    print(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', action='store_true', help='report binary success')
    parser.add_argument('--stdev', action='store_true')
    args = parser.parse_args()
    standard_error = False if args.stdev else True
    debug = False

    # --------------------------------------------------------------------------------------------- #
    # (1) Plot performance, one figure per demonstration count.
    # --------------------------------------------------------------------------------------------- #
    demos = [1, 10, 100, 1000]
    for d in demos:
        print('-'*200)
        print(f'Plotting results for {d} training demos.')
        plot_combo_fixed_demos(args, num_demos=d, standard_error=standard_error, debug=debug)
        print('-'*200)

    # --------------------------------------------------------------------------------------------- #
    # (2) Print LaTeX code so I can put it into the paper.
    # --------------------------------------------------------------------------------------------- #
    print('\n')
    print('-'*200)
    print('-'*200)
    print('-'*200)
    print_table(args)

    # --------------------------------------------------------------------------------------------- #
    # (3) A single task/table.
    # --------------------------------------------------------------------------------------------- #
    ## tasks_l = ['cable-ring', 'cable-ring-notarget', 'cable-shape', 'cloth-cover',
    ##            'cloth-flat', 'bag-alone-open', 'bag-items-easy', 'bag-items-hard',
    ##            'cable-line-notarget', 'cable-shape-notarget', 'cloth-flat-notarget',
    ##            'insertion-goal', 'bag-color-goal']

    tasks_l = ['insertion-goal']

    for name_task in tasks_l:
        if name_task in REPLACE_TASK:
            name_plot = REPLACE_TASK[name_task]
        else:
            name_plot = name_task
        goal_conditioned = name_task in GOAL_CONDITIONED_TASKS
        print('\n')
        print('-'*200)
        print('-'*200)
        print('-'*200)
        print_single(args, goal_conditioned, name_task=name_task, name_plot=name_plot)
        plot_single(args, goal_conditioned, name_task=name_task, name_plot=name_plot)