# Example O3 Configuration
# This file shows how to customize O3 for your specific needs

# Custom model configurations
CUSTOM_MODEL_CONFIGS = {
    "my-model:7b": {
        "num_ctx": [4096, 8192, 16384],
        "batch": [16, 32],
        "f16_kv": [True],
        "num_predict": [256, 512]
    },
    "my-large-model:30b": {
        "num_ctx": [4096, 8192, 12288],  # Smaller range for large models
        "batch": [8, 16],                # Smaller batches for VRAM constraints
        "f16_kv": [True, False],         # Test both precisions
        "num_predict": [256]
    }
}

# Test settings
DEFAULT_CONCURRENCY_LEVELS = [1, 2]
DEFAULT_REPETITIONS = 3
DEFAULT_TIMEOUT = 90  # seconds

# Hardware-specific settings
AMD_GPU_SETTINGS = {
    "monitor_command": "rocm-smi --showmemuse --csv"
}

NVIDIA_GPU_SETTINGS = {
    "monitor_command": "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
}

# Output settings
OUTPUT_FORMATS = ["jsonl", "yaml", "csv"]
GENERATE_PLOTS = False  # Set to True if matplotlib available

# Safety settings
MAX_VRAM_USAGE_PERCENT = 90
MAX_RAM_USAGE_PERCENT = 85
TEMPERATURE_THRESHOLD = 85  # Celsius
