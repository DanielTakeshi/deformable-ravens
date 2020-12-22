#!/usr/bin/env python

import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, linewidth=200)
from ravens import utils


# #-----------------------------------------------------------------------------
# # Specify Training Plots
# #-----------------------------------------------------------------------------

# title = 'Transporter Nets Performance on Various Tasks'
# # files = {'Sorting':     'sorting-transporter-1.pkl',
# #          'Insertion':   'insertion-transporter-1.pkl',
# #          'Hanoi':       'hanoi-transporter-1.pkl',
# #          'Aligning':    'aligning-transporter-1.pkl',
# #          'Stacking':    'stacking-transporter-1.pkl',
# #          'Sweeping':    'sweeping-transporter-1.pkl',
# #          'Pushing':     'pushing-transporter-1.pkl',
# #          'Palletizing': 'palletizing-transporter-1.pkl',
# #          'Kitting':     'kitting-transporter-1.pkl',
# #          'Packing':     'packing-transporter-1.pkl'}
# files = {'Sorting':     'sorting-transporter-10.pkl',
#          'Insertion':   'insertion-transporter-10.pkl',
#          'Hanoi':       'hanoi-transporter-10.pkl',
#          'Aligning':    'aligning-transporter-10.pkl',
#          'Stacking':    'stacking-transporter-10.pkl',
#          'Sweeping':    'sweeping-transporter-10.pkl',
#          # 'Pushing':     'pushing-transporter-10.pkl',
#          'Palletizing': 'palletizing-transporter-10.pkl',
#          'Kitting':     'kitting-transporter-10.pkl',
#          'Packing':     'packing-transporter-10.pkl'}
# ylabel = 'Task Success (%)'
# xlabel = 'Training Steps'

# #-----------------------------------------------------------------------------
# # Generate Training Plots
# #-----------------------------------------------------------------------------

# logs = {}
# for name, file in files.items():
#     if os.path.isfile(file):
#         data = pickle.load(open(file, 'rb'))
#         data = np.float32(data)
#         x = np.sort(np.unique(data[:, 0]))
#         y = np.float32([data[data[:, 0] == ix, 1] for ix in x])
#         logs[name] = (x, y)
# fname = os.path.join(os.getcwd(), 'plot.png')
# utils.plot(fname, title, ylabel, xlabel, data=logs, ylim=[0, 1])
# print(f'Done. Plot image saved to: {fname}')

#-----------------------------------------------------------------------------
# Specify Sample Efficiency Plots
# (Daniel) need a special case to deal with my new file naming with cropping.
# Put an extra space so it's easy to detect with split().
#-----------------------------------------------------------------------------

title = 'Sample Efficiency on Various Tasks'
files = {
    #'Sorting-Crop-After':    'sorting-transporter- -rots-24-crop_bef_q-0',
    #'Sorting-Crop-Before':   'sorting-transporter- -rots-24-crop_bef_q-1',
    #'Insertion-Crop-After':  'insertion-transporter- -rots-24-crop_bef_q-0',
    #'Insertion-Crop-Before': 'insertion-transporter- -rots-24-crop_bef_q-1',
    #'Sorting':     'sorting-transporter-',
    #'Insertion':   'insertion-transporter-',
    #'Hanoi':       'hanoi-transporter-',
    #'Aligning':    'aligning-transporter-',
    #'Stacking':    'stacking-transporter-',
    #'Sweeping':    'sweeping-transporter-',
    #'Cable':       'cable-transporter-',
    #'Palletizing': 'palletizing-transporter-',
    #'Kitting':     'kitting-transporter-',
    #'Packing':     'packing-transporter-'
    'Insertion-GtState':       'insertion-gt_state-',
    'Insertion-GtState2Step':  'insertion-gt_state_2_step-',
}
ylabel = 'Task Success (%)'
xlabel = '# of Demonstrations'

#-----------------------------------------------------------------------------
# Generate Training Plots. After loading from pickle file and converting to a
# numpy array, `data` has shape (N,2) where the first item contains the
# training itr. Has structure like this if we did 20 test episodes each itr for
# 20K itrs:
#
# data = [
#   [01000, result_01],
#    ...
#   [01000, result_20],
#   [02000, result_01],
#    ...
#   [02000, result_20],
#    ...
#    ...
#   [20000, result_20],
# ]
#
# We then use np.unique() to get only unique itr values (usually there are
# multiple episodes per itr). Usually this would be every multiple of 1000, up
# to 20K. For each itr value, we extract all the results that match it, and
# then arrange into `iy` which is a list of numpy arrays (one for each value in
# `ix`, or itr). Then take the last 5 of these, corresponding to the last 5 itr
# values (16K, 17K, 18K, 19K, 20K).
# -----------------------------------------------------------------------------
# (Aug 12) Make sure the *.pkl files are in the `test_results/` directory.
# -----------------------------------------------------------------------------
# (Oct 11) Now working for gt_state and gt_state_2_step baselines, for the tasks
# where we actually test during training. For all of my stuff, we should instead
# use plot_defs.py or similar scripts, but keep this since it's helpful for
# testing with tasks tested in the CoRL paper.
# -----------------------------------------------------------------------------
# (Oct 12) Just remembered, we actually have the last 5 test runs averaged, but
# this is with snapshots {12K, 14K, 16K, 18K, 20K} whereas the transporters paper
# used {16K, 17K, 18K, 19K, 20K}. We can just use the last three. Update: well,
# actually for training insertion with main.py we actually did test every 1K
# snapshots ... but if we use load.py, we only did every 2K, argh confusing.
# -----------------------------------------------------------------------------

train_seed = 0
logs = {}
FILE_HEAD = 'test_results_insertion'  # Daniel: ADJUST IF NECESSARY.
LAST_FEW_RUNS = 5  # Transporters paper did 5

for name, file in files.items():
    print(f'\nPlotting: {name}, {file}')
    x, y, std = [0], [0], [0]
    for order in range(4):
        if 'crop_befe_q' in file:
            file1, file2 = file.split()
            fname = file1 + f'{10**order}-{train_seed}' + file2 + '.pkl'
        else:
            fname = file + f'{10**order}-{train_seed}.pkl'
        fname = os.path.join(FILE_HEAD, fname)
        print(f'fname: {fname}')
        if os.path.isfile(fname):
            x.append(len(x))
            data = pickle.load(open(fname, 'rb'))
            data = np.float32(data)
            ix = np.sort(np.unique(data[:, 0]))
            iy = np.float32([data[data[:, 0] == i, 1] for i in ix])
            iy = iy[-LAST_FEW_RUNS:].reshape(-1)  # see comments above
            std.append(np.std(iy))
            y.append(np.mean(iy))
            print(f'  {np.mean(iy):0.3f} +/- {np.std(iy):0.3f}, for {fname}')
    logs[name] = (x, y, std)
xticks = ['0', '1', '10', '100', '1000']
fname = os.path.join(os.getcwd(), 'figures', f'plot_train_seed_{train_seed}.png')
utils.plot(fname, title, ylabel, xlabel, data=logs, xticks=xticks, ylim=[0, 1])
print(f'Done. Plot image saved to: {fname}')
