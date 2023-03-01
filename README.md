**NOTE**: this code is based heavily on [the Ravens code base from Google][6]
and retains the same license.

# DeformableRavens

Code for the ICRA 2021 paper *Learning to Rearrange Deformable Cables, Fabrics, and Bags
with Goal-Conditioned Transporter Networks*. [Here is the project website][5],
which also contains the data we used to train policies.  Contents of this
README:

- [Installation](#installation)
- [Environments and Tasks](#environments-and-tasks)
- [Code Usage](#code-usage)
- [Downloading the Data](#downloading-the-data)
- [Miscellaneous](#miscellaneous)


## Installation

This is how to get the code running on a local machine. First, get conda on the
machine if it isn't there already:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then, create a new Python 3.7 conda environment (e.g., named "py3-defs") and
activate it:

```
conda create -n py3-defs python=3.7
conda activate py3-defs
```

Then install:

```
./install_python_ubuntu.sh
```

**Note I**: It is tested on **Ubuntu 18.04**. We have not tried other Ubuntu
versions or other operating systems.

**Note II**: Installing TensorFlow using conda is usually easier than pip
because the conda version will ship with the correct CUDA and cuDNN libraries,
whereas the pip version is a nightmare regarding version compatibility.

**Note III**: the code has only been tested with **PyBullet 3.0.4**. In fact,
there are some places which explicitly hard-code this requirement. Using later
versions may work but is not recommended.

## Environments and Tasks

This repository contains tasks in the ICRA 2021 paper and the predecessor
paper on Transporters (presented at CoRL 2020). For the latter paper, there are
(roughly) 10 tasks that came pre-shipped; the Transporters paper doesn't test
with `pushing` or `insertion-translation`, but tests with all others. See
`Tasks.md` for some task-specific documentation

Each task subclasses a `Task` class and needs to define its own `reset()`. The
`Task` class defines an oracle policy that's used to get demonstrations (so it
is not implemented within each task subclass), and is divided into cases
depending on the action, or `self.primitive`, used.

Similarly, different tasks have different reward functions, but all are
integrated into the `Task` super-class and divided based on the `self.metric`
type: `pose` or `zone`.

## Code Usage

Experiments start with `python main.py`, with `--disp` added for seeing the
PyBullet GUI (but not used for large-scale experiments). The general logic for
`main.py` proceeds as follows:

- Gather expert demonstrations for the task and put it in `data/{TASK}`, unless
  there are already a sufficient amount of demonstrations. There are
  sub-directories for `action`, `color`, `depth`, `info`, etc., which store the
  data pickle files with consistent indexing per time step. **Caution**: this
  will start "counting" the data from the existing `data/` directory. If you
  want entirely fresh data, delete the relevant file in `data/`.

- Given the data, train the designated agent. The logged data is stored in
  `logs/{AGENT}/{TASK}/{DATE}/{train}/` in the form of a `tfevent` [file for
  TensorBoard][4]. **Note**: it will do multiple training runs for statistical
  significance.

For deformables, we actually use a separate `load.py` script, due to some
issues with creating multiple environments.

**See `Commands.md` for commands to reproduce experimental results.**

## Downloading the Data

We normally generate 1000 demos for each of the tasks. However, this can take a
long time, especially for the bag tasks. We have pre-generated datasets for all
the tasks we tested with [on the project website][5]. Here's how to do this.
For example, suppose we want to download demonstration data for the
"bag-color-goal" task. Download the demonstration data from the website. Since
this is also a goal-conditioned task, download the *goal* demonstrations as
well. Make new `data/` and `goals/` directories and put the tar.gz files in the
respective directories:

```
deformable-ravens/
    data/
        bag-color-goal_1000_demos_480Hz_filtered_Nov13.tar.gz
    goals/
        bag-color-goal_20_goals_480Hz_Nov19.tar.gz
```

*Note*: if you generate data using the `main.py` script, then it will
automatically create the `data/` scripts, and similarly for the
`generate_goals.py` script. You only need to manually create `data/` and
`goals/` if you only want to download and get pre-existing datasets in the
right spot.

Then untar both of them in their respective directories:

```
tar -zxvf bag-color-goal_1000_demos_480Hz_filtered_Nov13.tar.gz
tar -zxvf bag-color-goal_20_goals_480Hz_Nov19.tar.gz
```

Now the data should be ready! If you want to inspect and debug the data, for
example the goals data, then do:

```
python ravens/dataset.py --path goals/bag-color-goal/
```

Note that by default it saves any content in `goals/` to `goals_out/` and data
in `data/` to `data_out/`. Also, by default, it will download and save images.
This can be very computationally intensive if you do this for the full 1000
demos. (The `goals/` data only has 20 demos.) You can change this easily in the
main method of `ravens/datasets.py`.

Running the script will print out some interesting data statistics for you.


## Miscellaneous

If you have questions, please use the public issue tracker.

If you find this code or research paper helpful, please consider citing it:

```
@inproceedings{seita_bags_2021,
    author    = {Daniel Seita and Pete Florence and Jonathan Tompson and Erwin Coumans and Vikas Sindhwani and Ken Goldberg and Andy Zeng},
    title     = {{Learning to Rearrange Deformable Cables, Fabrics, and Bags with Goal-Conditioned Transporter Networks}},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    Year      = {2021}
}
```

[1]:https://www.tensorflow.org/hub/installation
[2]:https://github.com/tensorflow/addons/issues/1132
[3]:https://partner-code.googlesource.com/project-reach/+/75459a560ea9ae4b9d7283ef39d4a4d99598ab81
[4]:https://stackoverflow.com/a/56537286/3287820
[5]:https://berkeleyautomation.github.io/bags/
[6]:https://github.com/google-research/ravens
