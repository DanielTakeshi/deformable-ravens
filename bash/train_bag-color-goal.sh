TASK=bag-color-goal
AGENT=transporter-goal      # transporter-goal, transporter-goal-(S)naive, gt_state, gt_state_2_step

python main.py --gpu=0 --gpu_mem_limit=12 --task=${TASK} --agent=${AGENT} --num_demos=1
python main.py --gpu=0 --gpu_mem_limit=12 --task=${TASK} --agent=${AGENT} --num_demos=10
python main.py --gpu=0 --gpu_mem_limit=12 --task=${TASK} --agent=${AGENT} --num_demos=100
python main.py --gpu=0 --gpu_mem_limit=12 --task=${TASK} --agent=${AGENT} --num_demos=1000
