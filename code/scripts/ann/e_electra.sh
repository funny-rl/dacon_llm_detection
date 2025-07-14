#!/bin/bash

torchrun --nproc_per_node=1 _ann_eval.py\
    --model_name "kykim/electra-kor-base"\
