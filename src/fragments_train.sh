#!/usr/bin/bash
source activate tensorflow

fragments=(100 500 1000 3000)

for n in ${fragments[@]}; do
    echo "#training $n bp files"
    python train.py --input_dim=$n
done