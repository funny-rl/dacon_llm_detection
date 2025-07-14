#!/bin/bash

torchrun --nproc_per_node=4 _ann.py\
    --model_name "klue/roberta-large"\
    --max_length 512\
    --split_ratio 0.8\
