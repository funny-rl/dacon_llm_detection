#!/bin/bash

torchrun --nproc_per_node=4 _ann.py\
    --model_name "klue/roberta-large"\
    --split_ratio 1.0\
