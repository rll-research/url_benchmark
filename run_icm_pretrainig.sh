#!/bin/bash
echo "in script"

eval "$(conda shell.bash hook)"
conda activate urlb2
if [ $1 = "pretrain" ]; then
	python pretrain.py group_name=$2 agent=$3 domain=$4 seed=$5
else
	python finetune.py group_name=$2 pretrained_agent=$3 task=$4 snapshot_ts=$5 obs_type=states seed=$6
fi

