#!/bin/bash

# Ensure API_KEY is set
if [ -z "$API_KEY" ]; then
  echo "Error: API_KEY environment variable is not set."
  echo "Please run: export API_KEY=your_key_here"
  exit 1
fi

python LLMood.py \
  --api_type openai \
  --api_key "$API_KEY" \
  --model gpt-4o-2024-08-06 \
  --temperature 0.5 \
  --parallel 5 \
  --experiment-type experiment1 \
  --moods Neutral Fear Anxiety Anger Disgust Sadness Worry \
  --experiment_name Exp1-gpt4o-2024-08-06 \
  --iterations 5 \
  -v


