#!/bin/bash expect -f

FILEPATH="./t5_predictions.txt"
OUTPATH="./t5-base-lm-adapt-eval-fos-9500"
INFILE="./instructions_sample.jsonl"

python ./evaluation_scripts/metrics_eval_t5.py $FILEPATH $OUTPATH $INFILE





