# Mainly using this for a sanity check to ensure that gt_state and gt_state_2_step are working.
# And also transporters, I guess?

TASK=insertion
AGENT=gt_state_2_step     # transporter, gt_state, gt_state_2_step

# To train:

# python main.py --gpu=0 --gpu_mem_limit=4 --task=${TASK} --agent=${AGENT} --num_demos=1
# python main.py --gpu=0 --gpu_mem_limit=4 --task=${TASK} --agent=${AGENT} --num_demos=10
# python main.py --gpu=0 --gpu_mem_limit=4 --task=${TASK} --agent=${AGENT} --num_demos=100
# python main.py --gpu=0 --gpu_mem_limit=4 --task=${TASK} --agent=${AGENT} --num_demos=1000

# To load:

for tr in 0 1 2; do
    python load.py --gpu=0 --gpu_mem_limit=4 --task=${TASK} --agent=${AGENT} --num_rots_inf=24 --train_run=${tr} --num_demos=1
    python load.py --gpu=0 --gpu_mem_limit=4 --task=${TASK} --agent=${AGENT} --num_rots_inf=24 --train_run=${tr} --num_demos=10
    python load.py --gpu=0 --gpu_mem_limit=4 --task=${TASK} --agent=${AGENT} --num_rots_inf=24 --train_run=${tr} --num_demos=100
    python load.py --gpu=0 --gpu_mem_limit=4 --task=${TASK} --agent=${AGENT} --num_rots_inf=24 --train_run=${tr} --num_demos=1000
done

# Note: loading will already happen in the main.py script, but the above load.py will do it separately,
# and ensures that the updated loading code I have is actually functional, so that it's OK to use for
# the "real" gt_state(_2_step) tasks.
