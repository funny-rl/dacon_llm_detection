#!/bin/bash

torchrun --nproc_per_node=1 _ann.py\
    --model_name "kykim/electra-kor-base"\
    --split_ratio 1.0\
