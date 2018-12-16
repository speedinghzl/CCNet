#!/bin/bash
uname -a
date

CS_PATH=$1
LR=1e-2
WD=1e-4
BS=8
R=2
INPUT_SIZE=769
STEPS=60000
GPU_IDS=0,1,2,3
OHEM=1 #set to 1 for reducing the performance gap between val and test set.

#variable ${LOCAL_OUTPUT} dir can save data of you job, after exec it will be upload to hadoop_out path 
python train.py --data-dir ${CS_PATH} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu ${GPU_IDS} --learning-rate ${LR} --input-size ${INPUT_SIZE},${INPUT_SIZE} --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS} --recurrence ${R} --ohem ${OHEM} --ohem-thres 0.7 --ohem-keep 100000
python evaluate.py --data-dir ${CS_PATH} --restore-from snapshots/CS_scenes_${STEPS}.pth --gpu 0 --recurrence ${R}
