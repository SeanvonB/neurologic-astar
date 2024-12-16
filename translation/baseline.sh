#!/usr/bin/env bash

export PYTHONPATH="C:\Users\Sean\Documents\NLP Dump\NeuroLogic A-Star"

DEVICES=0
DATA_DIR="./dataset/machine_translation"

# run decoding
CUDA_VISIBLE_DEVICES=${DEVICES} python ./translation/baseline.py --model_name Helsinki-NLP/opus-mt-en-de \
  --input_file ${DATA_DIR}/newstest2017-iate/iate.414.terminology.tsv.en --output_file ./results/translation_baseline.txt \
  --batch_size 64 --beam_size 15 --max_tgt_length 156 --min_tgt_length 3 \
  --length_penalty 0.6


