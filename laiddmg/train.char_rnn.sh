#!/bin/bash

CUDA_VISIBLE_DEVICES=0

model=char_rnn
echo $model

laiddmg-train $model --seed 219 \
                     --output_dir exp1 \
                     --dataset_path ../datasets \
                     --log_steps 10 \
                     --num_train_epochs 100 \
                     --train_batch_size 256 \
                     --lr 1e-3 \
                     --step_size 10 \
                     --gamma 0.5

                     #--train_batch_size 128 \
