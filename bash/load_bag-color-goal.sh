AGENT=transporter-goal      # transporter-goal, transporter-goal-naive, gt_state, gt_state_2_step
TASK=bag-color-goal
HZ=480
ROTS=1

# This is one training run, set at TR. We did TR=0,1,2, hopefully!
TR=0
python load.py --gpu=0 --gpu_mem_limit=12 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${TR} --num_demos=1
python load.py --gpu=0 --gpu_mem_limit=12 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${TR} --num_demos=10
python load.py --gpu=0 --gpu_mem_limit=12 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${TR} --num_demos=100
python load.py --gpu=0 --gpu_mem_limit=12 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${TR} --num_demos=1000
