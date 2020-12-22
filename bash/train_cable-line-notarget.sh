TASK=cable-line-notarget        # cable-shape-notarget, cable-line-notarget, insertion-goal, cloth-flat-notarget
AGENT=transporter-goal-naive    # transporter-goal, transporter-goal-naive, ground truth?

python main.py --gpu=0 --gpu_mem_limit=10 --task=${TASK} --agent=${AGENT} --num_demos=1
python main.py --gpu=0 --gpu_mem_limit=10 --task=${TASK} --agent=${AGENT} --num_demos=10
python main.py --gpu=0 --gpu_mem_limit=10 --task=${TASK} --agent=${AGENT} --num_demos=100
python main.py --gpu=0 --gpu_mem_limit=10 --task=${TASK} --agent=${AGENT} --num_demos=1000
