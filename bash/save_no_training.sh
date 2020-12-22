# Note: for these we probably don't need to do the demos thing, we can just copy to all others.

# Non-goal conditioned.

AGENT=transporter
for TASK in cable-shape cable-ring cable-ring-notarget cloth-cover cloth-flat bag-alone-open bag-items-easy bag-items-hard ; do
    python main.py --save_zero --gpu=0 --gpu_mem_limit=12 --task=${TASK} --agent=${AGENT} --num_demos=1
done

AGENT=gt_state
for TASK in cable-shape cable-ring cable-ring-notarget cloth-cover cloth-flat bag-alone-open bag-items-easy bag-items-hard ; do
    python main.py --save_zero --gpu=0 --gpu_mem_limit=3 --task=${TASK} --agent=${AGENT} --num_demos=1
done

AGENT=gt_state_2_step
for TASK in cable-shape cable-ring cable-ring-notarget cloth-cover cloth-flat bag-alone-open bag-items-easy bag-items-hard ; do
    python main.py --save_zero --gpu=0 --gpu_mem_limit=3 --task=${TASK} --agent=${AGENT} --num_demos=1
done
 

# Goal-conditioned.

AGENT=transporter-goal
for TASK in cable-shape-notarget cable-line-notarget cloth-flat-notarget bag-color-goal insertion-goal ; do
    python main.py --save_zero --gpu=0 --gpu_mem_limit=12 --task=${TASK} --agent=${AGENT} --num_demos=1
done

AGENT=transporter-goal-snaive
for TASK in cable-shape-notarget cable-line-notarget cloth-flat-notarget bag-color-goal insertion-goal ; do
    python main.py --save_zero --gpu=0 --gpu_mem_limit=12 --task=${TASK} --agent=${AGENT} --num_demos=1
done

AGENT=gt_state
for TASK in cable-shape-notarget cable-line-notarget cloth-flat-notarget bag-color-goal insertion-goal ; do
    python main.py --save_zero --gpu=0 --gpu_mem_limit=3 --task=${TASK} --agent=${AGENT} --num_demos=1
done

AGENT=gt_state_2_step
for TASK in cable-shape-notarget cable-line-notarget cloth-flat-notarget bag-color-goal insertion-goal ; do
    python main.py --save_zero --gpu=0 --gpu_mem_limit=3 --task=${TASK} --agent=${AGENT} --num_demos=1
done
