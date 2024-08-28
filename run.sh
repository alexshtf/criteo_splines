#!/bin/sh

emb_dim=48
gpus=(3 4 5)
session_name=splines_${emb_dim}
n_epochs=10
common_args="--n_epochs ${n_epochs}  --emb_dims ${emb_dim}"

tmux new-session -d -s $session_name \
    "python3 train.py ${common_args} --gpu ${gpus[0]} --study_name criteo_bins_${emb_dim}"
tmux split-window -v \
    "python3 train.py ${common_args} --gpu ${gpus[1]} --splines True --degrees 3 --study_name criteo_splines_3_${emb_dim}"
tmux split-window -v \
    "python3 train.py ${common_args} --gpu ${gpus[2]} --splines True --degrees 0 --study_name criteo_splines_0_${emb_dim}"
tmux split-window -hf "nvitop"

tmux attach-session -t $session_name