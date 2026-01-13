#!/bin/bash
# ================================================
# Qwen3-8B GRPO Agentic Training Script
# Backend: DeepSpeed, Algorithm: GRPO
# ================================================

set +x

# Get the configuration directory name
CONFIG_PATH=$(basename $(dirname $0))

# Run the agentic pipeline with the GRPO configuration
python examples/start_agentic_pipeline.py \
    --config_path $CONFIG_PATH \
    --config_name grpo_agentic_config
