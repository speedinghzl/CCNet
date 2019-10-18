#!/bin/bash
uname -a
date

CS_PATH=/home/v-wesu/data/cityscapes/cityscapes.zip@/cityscapes
RESTORE_FROM=/home/v-wesu/data/Pretrained_Model/resnet101-imagenet.pth
SNAPSHOT_HOME=/home/v-wesu/data/CCNet_SnapShots
MODEL=ccnet
LR=1e-2
WD=5e-4
BS=8
STEPS=60000
INPUT_SIZE=769, 769
OHEM=0
GPU_IDS=0,1,2,3

#variable ${LOCAL_OUTPUT} dir can save data of you job, after exec it will be upload to hadoop_out path 
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 train.py \
  --data-dir ${CS_PATH} \
  --model ${MODEL} \
  --random-mirror \
  --random-scale \
  --restore-from ${RESTORE_FROM} \
  --input-size ${INPUT_SIZE} \
  --gpu ${GPU_IDS} \
  --learning-rate ${LR}  \
  --weight-decay ${WD} \
  --batch-size ${BS} \
  --num-steps ${STEPS} \
  --ohem ${OHEM} \
  --snapshot-dir ${SNAPSHOT_HOME}/${MODEL}

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 evaluate.py \
  --data-dir ${CS_PATH} \
  --model ${MODEL} \
  --input-size ${INPUT_SIZE} \
  --batch-size 4 \
  --restore-from ${SNAPSHOT_HOME}/${MODEL}/CS_scenes_${STEPS}.pth \
  --gpu 0
