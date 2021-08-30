#!/bin/bash

model=char_rnn
# model=vae
# echo $model

laiddmg-generate $model --seed 219 \
                        --checkpoint_dir ./outputs/$model/exp1 \
                        --weights_name ckpt_010.pt \
                        --num_generation 10000 \
                        --batch_size_for_generation 256
