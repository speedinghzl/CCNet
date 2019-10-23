#!/bin/bash
uname -a
date

CS_PATH=/blob/data/cityscapes/cityscapes.zip@/cityscapes
RESTORE_FROM=/blob/data/model/resnet101-imagenet.pth
SNAPSHOT_HOME=/blob/data/CCNet/snapshot
LR=1e-2
WD=5e-4
BS=4
STEPS=60000
OHEM=0
GPU_IDS=0,1

Judgement=train
MODEL=baseline
INPUT_S=769
INPUT_SIZE=${INPUT_S},${INPUT_S}

if [[ ${Judgement} == 'train' ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=2 train.py \
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
    --snapshot-dir ${SNAPSHOT_HOME}/${MODEL}_${INPUT_S}
fi

if [[ ${Judgement} == 'evaluate' ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 evaluate.py \
    --data-dir ${CS_PATH} \
    --model ${MODEL} \
    --input-size ${INPUT_SIZE} \
    --batch-size 4 \
    --restore-from ${SNAPSHOT_HOME}/${MODEL}_${INPUT_S}/CS_scenes_${STEPS}.pth \
    --gpu 0
fi