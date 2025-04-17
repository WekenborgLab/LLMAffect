#!/bin/bash

# Ensure API_KEY is set
if [ -z "$API_KEY" ]; then
  echo "Error: API_KEY environment variable is not set."
  echo "Please run: export API_KEY=your_key_here"
  exit 1
fi

python LLMood.py \
  --api_type vllm \
  --api_key "$API_KEY" \
  --api_base http://192.168.33.30:8001/v1/ \
  --model Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --temperature 0.5 \
  --parallel 5 \
  --experiment-type experiment1 \
  --moods Neutral Fear Anxiety Anger Disgust Sadness Worry \
  --experiment_name Exp1-Llama4-Maverick \
  --iterations 5 \
  -v

python LLMood.py \
  --api_type vllm \
  --api_key "$API_KEY" \
  --api_base http://192.168.33.30:8000/v1/ \
  --model Llama-4-Scout-17B-16E-Instruct \
  --temperature 0.5 \
  --parallel 5 \
  --experiment-type experiment1 \
  --moods Neutral Fear Anxiety Anger Disgust Sadness Worry \
  --experiment_name Exp1-Llama4-Scout \
  --iterations 5 \
  -v

python LLMood.py \
  --api_type vllm \
  --api_key "$API_KEY" \
  --api_base http://192.168.33.30:8002/v1/ \
  --model gpt-4 \
  --temperature 0.5 \
  --parallel 5 \
  --experiment-type experiment1 \
  --moods Neutral Fear Anxiety Anger Disgust Sadness Worry \
  --experiment_name Exp1-Qwen2.5-72B-VL \
  --iterations 5 \
  -v

python LLMood.py \
  --api_type vllm \
  --api_key "$API_KEY" \
  --api_base http://192.168.33.30:8003/v1/ \
  --model Llama-3.3-70B-Instruct \
  --temperature 0.5 \
  --parallel 5 \
  --experiment-type experiment1 \
  --moods Neutral Fear Anxiety Anger Disgust Sadness Worry \
  --experiment_name Exp1-Llama-3.3-70B-Instruct-VLLM \
  --iterations 5 \
  -v

python LLMood.py \
  --api_type vllm \
  --api_key "$API_KEY" \
  --api_base http://192.168.33.30:8004/v1/ \
  --model Llama-3.1-8B-Instruct \
  --temperature 0.5 \
  --parallel 5 \
  --experiment-type experiment1 \
  --moods Neutral Fear Anxiety Anger Disgust Sadness Worry \
  --experiment_name Exp1-Llama-3.1-8B-Instruct-VLLM \
  --iterations 5 \
  -v
