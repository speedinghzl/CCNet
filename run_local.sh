#!/bin/bash
uname -a
#date
#env
date

CS_PATH=$1
MODEL=$2
LR=1e-2
WD=5e-4
BS=8
STEPS=$3
INPUT_SIZE=$4
OHEM=$5
GPU_IDS=0,1,2,3

#variable ${LOCAL_OUTPUT} dir can save data of you job, after exec it will be upload to hadoop_out path 
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --input-size ${INPUT_SIZE} --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS} --ohem ${OHEM}
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS}
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 evaluate.py --data-dir ${CS_PATH} --model ${MODEL} --input-size ${INPUT_SIZE} --batch-size 4 --restore-from snapshots/CS_scenes_${STEPS}.pth --gpu 0
