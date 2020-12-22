TASK=insertion-goal             # cable-shape-notarget, cable-line-notarget, insertion-goal, cloth-flat-notarget
AGENT=gt_state                  # transporter-goal, transporter-goal-naive, gt_state, gt_state_2_step

python main.py --gpu=0 --gpu_mem_limit=3 --task=${TASK} --agent=${AGENT} --num_demos=1
python main.py --gpu=0 --gpu_mem_limit=3 --task=${TASK} --agent=${AGENT} --num_demos=10
python main.py --gpu=0 --gpu_mem_limit=3 --task=${TASK} --agent=${AGENT} --num_demos=100
python main.py --gpu=0 --gpu_mem_limit=3 --task=${TASK} --agent=${AGENT} --num_demos=1000
