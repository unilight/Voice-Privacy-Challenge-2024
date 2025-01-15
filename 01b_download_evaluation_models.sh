#!/bin/bash

set -e

source env.sh

for model in asv_orig; do
    if [ ! -d "exp/$model" ]; then
        if [ ! -f .${model}.zip ]; then
            echo "Download pretrained $model models pre-trained..."
            wget https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/pre_model.zip/${model}.zip
            mv ${model}.zip .${model}.zip
        fi
        echo "Unpacking pretrained evaluation models"
        unzip .${model}.zip
    fi
done
