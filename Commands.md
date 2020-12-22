# Commands to Run

Just so everything is here in one place, here are all the commands I use for
experiments, for data generation and training / loading policies, complete with
dates + commits used.

- Maybe keep 240Hz for the cables tasks?
- Definitely 480Hz for the fabrics + bags tasks.
- Make sure `main.py` will allow for 1000 demonstrations to be created.
  However, since my code will exit out of all of these, then we need one
  "extra" command to generate the data first.
- Generate data without camera noise (to start) and with camera noise for
  later.

# (1) Data Generation

For this the `agent` and `num_demos` don't matter. The Hz does matter, quite a
lot. I don't think the GPU matters but maybe it's OK to leave the argument
there. **After running 1K demos, save them as tar.gz files and store them**.
Save (a) the date, (b) num demos, (c) Hz value, and (d) whether there's noise
in camera images. For (d), this will probably not be a command line argument.

**Double check `data/` to make sure there is no data.**

Here are the cables data generation (+ insertion goal):

```
python main.py --gpu=0 --agent=dummy --hz=240 --task=cable-shape
python main.py --gpu=0 --agent=dummy --hz=240 --task=cable-ring
python main.py --gpu=0 --agent=dummy --hz=240 --task=cable-ring-notarget
python main.py --gpu=0 --agent=dummy --hz=240 --task=cable-shape-notarget
python main.py --gpu=0 --agent=dummy --hz=240 --task=cable-line-notarget
python main.py --gpu=0 --agent=dummy --hz=240 --task=insertion-goal
```

Then fabrics/ bags tasks:

```
python main.py --gpu=0 --agent=dummy --hz=480 --task=cloth-cover
python main.py --gpu=0 --agent=dummy --hz=480 --task=cloth-flat
python main.py --gpu=0 --agent=dummy --hz=480 --task=cloth-flat-notarget
python main.py --gpu=0 --agent=dummy --hz=480 --task=bag-alone-open
python main.py --gpu=0 --agent=dummy --hz=480 --task=bag-items-easy
python main.py --gpu=0 --agent=dummy --hz=480 --task=bag-items-hard
python main.py --gpu=0 --agent=dummy --hz=480 --task=bag-color-goal
```

# (1B) Generating Goals (if necessary) for loading goal-conditioned

This is pretty simple. However, be careful about Hz as usual.

```
python generate_goals.py --hz=240 --task=insertion-goal       --num_goals=20
python generate_goals.py --hz=240 --task=cable-shape-notarget --num_goals=20
python generate_goals.py --hz=240 --task=cable-line-notarget  --num_goals=20
python generate_goals.py --hz=480 --task=cloth-flat-notarget  --num_goals=20
python generate_goals.py --hz=480 --task=bag-color-goal       --num_goals=20
```

# (2) Training

Here the Hz does not matter. *However, the agent and GPU ID* definitely
matter. By default we do 20K iterations (saving snapshots every 1K iters) and
perform 3 training runs.

Also, for all tasks *other* than `bag-items-hard`, we really don't need the
extra rotations (24) but having those there should be harmless if all labels
have rotation 0, and including those rotations makes it helpful to understand
how rotations impact Transporters (e.g., with cloth).

If GPU memory is an issue (e.g., running into "Blas SGEMM launch failed"
errors) then use `--gpu_mem_limit` and set it to be less than the GB of RAM
that the target GPU has. The units of `--gpu_mem_limit` are in GB.

For baseline methods, use `gt_state` and `gt_state_2_step`. Any others?

### Non-goal conditioned

Can ablate over crop query ordering, which is 1 by default but we should
consider 0 as default as that's what goal conditioned Transporters does.

```
TASK=cable-shape   # cable-shape, cable-ring, cable-ring-notarget, cloth-cover, cloth-flat, bag-alone-open, bag-items-easy, bag-items-hard
AGENT=transporter  # transporter, gt_state, gt_state_2_step

python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --num_demos=1
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --num_demos=10
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --num_demos=100
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --num_demos=1000

python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --crop_bef_q=0 --num_demos=1
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --crop_bef_q=0 --num_demos=10
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --crop_bef_q=0 --num_demos=100
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --crop_bef_q=0 --num_demos=1000
```

### Goal-conditioned

We could ablate over `subsamp_g` to test different training procedures, but
probably not worth it.

```
TASK=cable-shape-notarget  # cable-shape-notarget, cable-line-notarget, insertion-goal, cloth-flat-notarget
AGENT=transporter-goal     # transporter-goal, transporter-goal-naive, gt_state, gt_state_2_step

python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --num_demos=1
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --num_demos=10
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --num_demos=100
python main.py --gpu=0 --task=${TASK} --agent=${AGENT} --num_demos=1000
```


# (3) Loading

Both the Hz and agent definitely matter, as well as the GPU if running on a
multi-GPU system. If we do 3x training runs, then use `train_run` to
distinguish among the seeds.

### Loading Cables (non goals)

```
HZ=240
ROTS=1
AGENT=transporter   # transporter, gt_state, gt_state_2_step
TASK=cable-shape    # cable-shape, cable-ring, cable-ring-notarget

for tr in 0 1 2; do
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=10
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=100
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1000
done

for tr in 0 1 2; do
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=1
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=10
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=100
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=1000
done
```

### Loading Cables + Insertion (with goals)

```
HZ=240
ROTS=1
AGENT=transporter-goal     # transporter-goal, transporter-goal-naive, gt_state, gt_state_2_step
TASK=cable-shape-notarget  # cable-shape-notarget, cable-line-notarget, insertion-goal

for tr in 0 1 2; do
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=10
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=100
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1000
done
```

### Loading Fabrics/Bags (non goals)

Now it's 480 Hz. **Note: if the task is `bag-items-hard` we should definitely
use 24 rotations (or however many we set during training). However, all other
tasks use 1 rotation during inference time**.

```
HZ=480
ROTS=1             # CHANGE TO 24 IF LOADING bag-items-hard
AGENT=transporter  # transporter, gt_state, gt_state_2_step
TASK=cloth-flat    # cloth-flat, cloth-cover bag-alone-open, bag-items-easy, bag-items-hard

for tr in 0 1 2; do
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=10
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=100
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1000
done

for tr in 0 1 2; do
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=1
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=10
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=100
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=1000
done
```

### Loading Fabrics/Bags (with goals)

Only `cloth-flat-target` and `bag-color-goal` apply here.

```
HZ=480
ROTS=1
AGENT=transporter-goal    # transporter-goal, transporter-goal-naive, gt_state, gt_state_2_step
TASK=cloth-flat-notarget  # bag-color-goal

for tr in 0 1 2; do
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=10
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=100
    python load.py --gpu=0 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1000
done
```
