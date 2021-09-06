#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=vae
echo $model

laiddmg-train $model --seed 219 \
                     --output_dir exp1 \
                     --dataset_path ../datasets \
                     --log_steps 10 \
                     --num_train_epochs 100 \
                     --train_batch_size 512 \
                     --lr 1e-4
