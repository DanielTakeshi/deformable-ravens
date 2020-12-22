# Copy over the 0 iteration results in a programmatic manner. SSH keys really help. :D
# For each task, and for each model type, we did one test-set rollout of 20 episodes.
# We did this for the "1 demo" case but it could apply to all because there was no training,
# so there's no distinction in performance between different dataset demos.

H1=pybullet-def-envs/test_results
H2=test_results
SS=snapshot-00000-eps-20.pkl
MACHINE=jensen.ist.berkeley.edu

#scp seita@${MACHINE}:${H1}/bag-alone-open-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-alone-open-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-alone-open-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-alone-open-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-alone-open-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-alone-open-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-alone-open-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-alone-open-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-alone-open-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-alone-open-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-alone-open-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-alone-open-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-alone-open-transporter-10-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-alone-open-transporter-100-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-alone-open-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-alone-open-transporter-1000-0-rots-24-crop_bef_q-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-color-goal-gt_state-1-0-rotsinf-01/${SS}                               ${H2}/bag-color-goal-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-gt_state-1-0-rotsinf-01/${SS}                               ${H2}/bag-color-goal-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-gt_state-1-0-rotsinf-01/${SS}                               ${H2}/bag-color-goal-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-gt_state-1-0-rotsinf-01/${SS}                               ${H2}/bag-color-goal-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-color-goal-gt_state_2_step-1-0-rotsinf-01/${SS}                        ${H2}/bag-color-goal-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-gt_state_2_step-1-0-rotsinf-01/${SS}                        ${H2}/bag-color-goal-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-gt_state_2_step-1-0-rotsinf-01/${SS}                        ${H2}/bag-color-goal-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-gt_state_2_step-1-0-rotsinf-01/${SS}                        ${H2}/bag-color-goal-gt_state_2_step-1000-0-rotsinf-01/
#
## Oops, realized we did 100-2 here, not 100-0, for Transporter-Goal-Split.
#scp seita@${MACHINE}:${H1}/bag-color-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}         ${H2}/bag-color-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}         ${H2}/bag-color-goal-transporter-goal-10-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}         ${H2}/bag-color-goal-transporter-goal-100-2-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}         ${H2}/bag-color-goal-transporter-goal-1000-0-rots-24-fin_g-rotsinf-01/
#
## Oops, realized we did 100-2 here, not 100-0, for Transporter-Goal-Stack.
#scp seita@${MACHINE}:${H1}/bag-color-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS}  ${H2}/bag-color-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS}  ${H2}/bag-color-goal-transporter-goal-snaive-10-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS}  ${H2}/bag-color-goal-transporter-goal-snaive-100-2-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-color-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS}  ${H2}/bag-color-goal-transporter-goal-snaive-1000-0-rots-24-fin_g-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-items-easy-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-items-easy-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-items-easy-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-items-easy-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-items-easy-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-items-easy-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-items-easy-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-items-easy-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-items-easy-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-items-easy-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-items-easy-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-items-easy-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-items-easy-transporter-10-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-items-easy-transporter-100-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-easy-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-items-easy-transporter-1000-0-rots-24-crop_bef_q-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-items-hard-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-items-hard-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-items-hard-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-items-hard-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-gt_state-1-0-rotsinf-01/${SS}                          ${H2}/bag-items-hard-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-items-hard-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-items-hard-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-items-hard-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-items-hard-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-gt_state_2_step-1-0-rotsinf-01/${SS}                   ${H2}/bag-items-hard-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/bag-items-hard-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-items-hard-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-items-hard-transporter-10-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-items-hard-transporter-100-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/bag-items-hard-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS}  ${H2}/bag-items-hard-transporter-1000-0-rots-24-crop_bef_q-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-line-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cable-line-notarget-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cable-line-notarget-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cable-line-notarget-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cable-line-notarget-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-line-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cable-line-notarget-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cable-line-notarget-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cable-line-notarget-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cable-line-notarget-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-line-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cable-line-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cable-line-notarget-transporter-goal-10-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cable-line-notarget-transporter-goal-100-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cable-line-notarget-transporter-goal-1000-0-rots-24-fin_g-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-line-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cable-line-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cable-line-notarget-transporter-goal-snaive-10-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cable-line-notarget-transporter-goal-snaive-100-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-line-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cable-line-notarget-transporter-goal-snaive-1000-0-rots-24-fin_g-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-ring-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-ring-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-ring-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-ring-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-ring-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-ring-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-ring-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-ring-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-ring-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-ring-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-ring-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-ring-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-ring-transporter-10-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-ring-transporter-100-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-ring-transporter-1000-0-rots-24-crop_bef_q-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-ring-notarget-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-ring-notarget-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-ring-notarget-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-ring-notarget-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-ring-notarget-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-ring-notarget-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-ring-notarget-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-ring-notarget-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-ring-notarget-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-ring-notarget-transporter-10-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-ring-notarget-transporter-100-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-ring-notarget-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-ring-notarget-transporter-1000-0-rots-24-crop_bef_q-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-shape-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-shape-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-shape-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-shape-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cable-shape-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-shape-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-shape-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-shape-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-shape-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cable-shape-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-shape-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-shape-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-shape-transporter-10-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-shape-transporter-100-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cable-shape-transporter-1000-0-rots-24-crop_bef_q-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cable-shape-notarget-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cable-shape-notarget-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cable-shape-notarget-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cable-shape-notarget-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cable-shape-notarget-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cable-shape-notarget-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cable-shape-notarget-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cable-shape-notarget-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cable-shape-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cable-shape-notarget-transporter-goal-10-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cable-shape-notarget-transporter-goal-100-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cable-shape-notarget-transporter-goal-1000-0-rots-24-fin_g-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cable-shape-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cable-shape-notarget-transporter-goal-snaive-10-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cable-shape-notarget-transporter-goal-snaive-100-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cable-shape-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cable-shape-notarget-transporter-goal-snaive-1000-0-rots-24-fin_g-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-cover-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cloth-cover-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cloth-cover-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cloth-cover-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cloth-cover-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-cover-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cloth-cover-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cloth-cover-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cloth-cover-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cloth-cover-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-cover-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cloth-cover-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cloth-cover-transporter-10-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cloth-cover-transporter-100-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-cover-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cloth-cover-transporter-1000-0-rots-24-crop_bef_q-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-flat-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cloth-flat-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cloth-flat-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cloth-flat-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-gt_state-1-0-rotsinf-01/${SS}                         ${H2}/cloth-flat-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-flat-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cloth-flat-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cloth-flat-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cloth-flat-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-gt_state_2_step-1-0-rotsinf-01/${SS}                  ${H2}/cloth-flat-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-flat-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cloth-flat-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cloth-flat-transporter-10-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cloth-flat-transporter-100-0-rots-24-crop_bef_q-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-transporter-1-0-rots-24-crop_bef_q-0-rotsinf-01/${SS} ${H2}/cloth-flat-transporter-1000-0-rots-24-crop_bef_q-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cloth-flat-notarget-gt_state-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cloth-flat-notarget-gt_state-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cloth-flat-notarget-gt_state-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-gt_state-1-0-rotsinf-01/${SS}                              ${H2}/cloth-flat-notarget-gt_state-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cloth-flat-notarget-gt_state_2_step-1-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cloth-flat-notarget-gt_state_2_step-10-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cloth-flat-notarget-gt_state_2_step-100-0-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-gt_state_2_step-1-0-rotsinf-01/${SS}                       ${H2}/cloth-flat-notarget-gt_state_2_step-1000-0-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cloth-flat-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cloth-flat-notarget-transporter-goal-10-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cloth-flat-notarget-transporter-goal-100-0-rots-24-fin_g-rotsinf-01/
#scp seita@${MACHINE}:${H1}/cloth-flat-notarget-transporter-goal-1-0-rots-24-fin_g-rotsinf-01/${SS}        ${H2}/cloth-flat-notarget-transporter-goal-1000-0-rots-24-fin_g-rotsinf-01/
#
# Oops, did this with 1 rot for training (not a big deal).
scp seita@${MACHINE}:${H1}/cloth-flat-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cloth-flat-notarget-transporter-goal-snaive-1-0-rots-1-fin_g-rotsinf-01/
scp seita@${MACHINE}:${H1}/cloth-flat-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cloth-flat-notarget-transporter-goal-snaive-10-0-rots-1-fin_g-rotsinf-01/
scp seita@${MACHINE}:${H1}/cloth-flat-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cloth-flat-notarget-transporter-goal-snaive-100-0-rots-1-fin_g-rotsinf-01/
scp seita@${MACHINE}:${H1}/cloth-flat-notarget-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-01/${SS} ${H2}/cloth-flat-notarget-transporter-goal-snaive-1000-0-rots-1-fin_g-rotsinf-01/
#
#scp seita@${MACHINE}:${H1}/insertion-goal-gt_state-1-0-rotsinf-24/${SS}                               ${H2}/insertion-goal-gt_state-1-0-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-gt_state-1-0-rotsinf-24/${SS}                               ${H2}/insertion-goal-gt_state-10-0-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-gt_state-1-0-rotsinf-24/${SS}                               ${H2}/insertion-goal-gt_state-100-0-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-gt_state-1-0-rotsinf-24/${SS}                               ${H2}/insertion-goal-gt_state-1000-0-rotsinf-24/
#
#scp seita@${MACHINE}:${H1}/insertion-goal-gt_state_2_step-1-0-rotsinf-24/${SS}                        ${H2}/insertion-goal-gt_state_2_step-1-0-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-gt_state_2_step-1-0-rotsinf-24/${SS}                        ${H2}/insertion-goal-gt_state_2_step-10-0-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-gt_state_2_step-1-0-rotsinf-24/${SS}                        ${H2}/insertion-goal-gt_state_2_step-100-0-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-gt_state_2_step-1-0-rotsinf-24/${SS}                        ${H2}/insertion-goal-gt_state_2_step-1000-0-rotsinf-24/
#
#scp seita@${MACHINE}:${H1}/insertion-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-24/${SS}         ${H2}/insertion-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-24/${SS}         ${H2}/insertion-goal-transporter-goal-10-0-rots-24-fin_g-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-24/${SS}         ${H2}/insertion-goal-transporter-goal-100-0-rots-24-fin_g-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-transporter-goal-1-0-rots-24-fin_g-rotsinf-24/${SS}         ${H2}/insertion-goal-transporter-goal-1000-0-rots-24-fin_g-rotsinf-24/
#
#scp seita@${MACHINE}:${H1}/insertion-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-24/${SS}  ${H2}/insertion-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-24/${SS}  ${H2}/insertion-goal-transporter-goal-snaive-10-0-rots-24-fin_g-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-24/${SS}  ${H2}/insertion-goal-transporter-goal-snaive-100-0-rots-24-fin_g-rotsinf-24/
#scp seita@${MACHINE}:${H1}/insertion-goal-transporter-goal-snaive-1-0-rots-24-fin_g-rotsinf-24/${SS}  ${H2}/insertion-goal-transporter-goal-snaive-1000-0-rots-24-fin_g-rotsinf-24/
