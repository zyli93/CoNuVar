#!/usr/bin/env bash

PYTHON=/home/ww8/anaconda3/bin/python3

$PYTHON conuvar.py -f data/exp1-2_counting_cnv.csv \
    --learning-rate 0.05 \
    --interval 10 \
    --sample-rate 0.003 \
    --num-proc 4 \
    --alpha 0.01 \
    --num-of-iteration 100
