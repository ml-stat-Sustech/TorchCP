#!/bin/bash

# 保存命令行参数到数组
SEEDS=(1 2 3 4 5)

# 循环执行命令
for SEED in "${SEEDS[@]}"
do
    python deepcp_examples/imagenet_standard.py --predictor Standard --seed $SEED
done
