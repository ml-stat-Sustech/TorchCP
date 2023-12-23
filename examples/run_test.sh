#!/bin/bash

predictors=("Standard" "ClassWise" "Cluster")
scores=("THR" "APS" "RAPS" "SAPS")

for predictor in "${predictors[@]}"
do
    for score in "${scores[@]}"
    do
        python examples/imagenet_example.py \
        --predictor ${predictor} \
        --score ${score} \
        --seed 0 
    done
done