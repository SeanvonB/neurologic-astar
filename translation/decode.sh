#!/usr/bin/env bash

export PYTHONPATH="PATH TO YOUR PROJECT ROOT DIR"

DATA_DIR="./dataset/machine_translation"
DATA_PREFIX="iate.414"
MODEL_NAME="Helsinki-NLP/opus-mt-en-de"

DEVICES=$1

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python ./translation/decode.py --model_name ${MODEL_NAME} \
  --input_path ${DATA_DIR}/newstest2017-iate/${DATA_PREFIX}.terminology.tsv.en --output_file ./results/translation_greedy.txt \
  --constraint_file ${DATA_DIR}/constraint/${DATA_PREFIX}.json \
  --batch_size 64 --beam_size 5 --max_tgt_length 156 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 200 --sat_tolerance 2 --beta 0.25 \
  --look_ahead_step 35  --alpha 0.05 --look_ahead_width 1 #--fusion_t 1.0

# neurologic with sampling look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python ./translation/decode.py --model_name ${MODEL_NAME} \
  --input_path ${DATA_DIR}/newstest2017-iate/${DATA_PREFIX}.terminology.tsv.en --output_file ./results/translation_sampling.txt \
  --constraint_file ${DATA_DIR}/constraint/${DATA_PREFIX}.json \
  --batch_size 64 --beam_size 5 --max_tgt_length 156 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 200 --sat_tolerance 2 --beta 0.25 \
  --look_ahead_step 35  --alpha 0.05  --look_ahead_sample --look_ahead_width 5

# neurologic with beam look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python ./translation/decode.py --model_name ${MODEL_NAME} \
  --input_path ${DATA_DIR}/newstest2017-iate/${DATA_PREFIX}.terminology.tsv.en --output_file ./results/translation_beam.txt \
  --constraint_file ${DATA_DIR}/constraint/${DATA_PREFIX}.json \
  --batch_size 64 --beam_size 5 --max_tgt_length 156 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 200 --sat_tolerance 2 --beta 0.25 \
  --look_ahead_step 35  --alpha 0.05 --look_ahead_width 2


