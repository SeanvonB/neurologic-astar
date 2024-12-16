#!/usr/bin/env bash

export PYTHONPATH="PATH TO YOUR PROJECT ROOT DIR"

DEVICES=$1
DATA_DIR="./dataset/question_generation"

# neurologic
CUDA_VISIBLE_DEVICES=${DEVICES} python ./question_generation/decode.py --model_name gpt2-large \
  --output_file ./results/qa_baseline.txt \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0 --look_ahead_width 1

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python ./question_generation/decode.py --model_name gpt2-large \
  --output_file ./results/qa_greedy.txt \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1 #--fusion_t 1.0

# neurologic with sample look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python ./question_generation/decode.py --model_name gpt2-large \
  --output_file ./results/qa_sample.txt \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_sample --look_ahead_width 2

# neurologic with beam look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python ./question_generation/decode.py --model_name gpt2-large \
  --output_file ./results/qa_beam.txt \
  --constraint_file ${DATA_DIR}/constraints.jsonl \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 2
