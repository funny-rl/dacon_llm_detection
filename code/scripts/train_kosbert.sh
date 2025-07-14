#!/bin/bash

torchrun --nproc_per_node=4 _ann.py\
    --model_name "jhgan/ko-sbert-nli"\
    --split_ratio 1.0\
