TASK=bag-items-easy
AGENT=transporter           # transporter, gt_state, gt_state_2_step
HZ=480
ROTS=1

# Train run is the training seed.
for tr in 0 1 2; do
    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=1
    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=10
    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=100
    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --crop_bef_q=0 --num_demos=1000
done

#for tr in 0 1 2; do
#    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1
#    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=10
#    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=100
#    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1000
#done

# Or this if we just want ONE train run (probably better). Comment out above.
#TR=0
#python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${TR} --crop_bef_q=0 --num_demos=1
#python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${TR} --crop_bef_q=0 --num_demos=10
#python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${TR} --crop_bef_q=0 --num_demos=100
#python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${TR} --crop_bef_q=0 --num_demos=1000
