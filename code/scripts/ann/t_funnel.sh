#!/bin/bash

torchrun --nproc_per_node=4 _ann.py\
    --model_name "kykim/funnel-kor-base"\
    --split_ratio 1.0\
