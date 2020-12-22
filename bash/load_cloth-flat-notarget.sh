AGENT=transporter-goal      # transporter-goal, transporter-goal-naive, TODO ground truth agent?
TASK=cloth-flat-notarget
HZ=480
ROTS=1

# Train run is the training seed.
for tr in 0 1 2; do
    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1
    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=10
    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=100
    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --num_demos=1000
done

#for tr in 0 1 2; do
#    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --subsamp-g --num_demos=1
#    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --subsamp-g --num_demos=10
#    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --subsamp-g --num_demos=100
#    python load.py --gpu=0 --gpu_mem_limit=14 --agent=${AGENT} --hz=${HZ} --task=${TASK} --num_rots_inf=${ROTS} --train_run=${tr} --subsamp-g --num_demos=1000
#done
