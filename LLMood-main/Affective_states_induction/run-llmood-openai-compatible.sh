#!/bin/bash

# Ensure API_KEY is set
if [ -z "$API_KEY" ]; then
  echo "Error: API_KEY environment variable is not set."
  echo "Please run: export API_KEY=your_key_here"
  exit 1
fi

python LLMood.py \
  --api_type openai-compatible \
  --api_key "$API_KEY" \
  --api_base http://pluto/v1/ \
  --model meta-llama-3.1-8b-instruct-q8 \
  --temperature 0.5 \
  --parallel 5 \
  --experiment-type experiment1 \
  --moods Neutral Fear Anxiety Anger Disgust Sadness Worry \
  --experiment_name Exp1-Llama3.1-8b \
  --iterations 5 \
  -v


python LLMood.py \
  --api_type openai-compatible \
  --api_key "$API_KEY" \
  --api_base http://pluto/v1/ \
  --model llama-3.3-70b-instruct-q4km \
  --temperature 0.5 \
  --parallel 2 \
  --experiment-type experiment1 \
  --moods Neutral Fear Anxiety Anger Disgust Sadness Worry \
  --experiment_name Exp1-Llama3.3-70b \
  --iterations 5 \
  -v
