#!/bin/bash

torchrun --nproc_per_node=1 _xgb.py\
    --model_name "jhgan/ko-sbert-nli"\
    --split_ratio 1.0\
