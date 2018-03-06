#!/usr/bin/env bash

PYTHON=/home/ww8/anaconda3/bin/python3

$PYTHON conuvar.py -f data/exp1-2_counting_cnv.csv -r 0.05 -t 10 -s 0.001 -p 4 -a 0.01 -n 100
