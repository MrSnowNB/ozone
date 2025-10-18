#!/bin/bash
echo "=== O3 Multi-GPU Test Setup for Codestral-22B ==="
echo "Hardware: 4x RTX A6000 (~186GB total VRAM)"

# Kill existing Ollama if running
pkill -f ollama

# Start Ollama with multi-GPU configuration
export OLLAMA_GPU_LAYERS=999
export OLLAMA_NUM_GPU=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OLLAMA_NUM_THREAD=48  # Multiple threads for multi-GPU

echo "Starting Ollama with multi-GPU configuration..."
ollama serve &
sleep 5

# Test the large model
echo "Testing Codestral-22B model with multi-GPU..."
ollama run codestral "Write a hello world script in Python" | head -10

echo "Multi-GPU test complete. If successful, run O3 optimization next."
