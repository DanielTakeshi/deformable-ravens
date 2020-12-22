TASK=cloth-flat    # cable-shape, cable-ring, cable-ring-notarget, cloth-cover, cloth-flat, bag-alone-open, bag-items-easy, bag-items-hard
AGENT=transporter  # transporter, TODO ground truth agent?

python main.py --gpu=0 --gpu_mem_limit=14 --task=${TASK} --agent=${AGENT} --crop_bef_q=0 --num_demos=1
python main.py --gpu=0 --gpu_mem_limit=14 --task=${TASK} --agent=${AGENT} --crop_bef_q=0 --num_demos=10
python main.py --gpu=0 --gpu_mem_limit=14 --task=${TASK} --agent=${AGENT} --crop_bef_q=0 --num_demos=100
python main.py --gpu=0 --gpu_mem_limit=14 --task=${TASK} --agent=${AGENT} --crop_bef_q=0 --num_demos=1000
