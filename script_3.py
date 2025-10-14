# Create comprehensive README.md
readme_content = '''# O3 (Ozone) - Ollama Open-Source Optimizer

![O3 Logo](https://img.shields.io/badge/O3-Ozone-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.7+-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Hardware-focused performance optimization suite for Ollama models, designed for agentic workflows and maximum context utilization.**

## 🎯 Purpose

O3 (Ozone) is a comprehensive testing suite that optimizes Ollama model configurations for:
- **Maximum stable context windows** for agentic workflows
- **Optimal throughput** (tokens/second) on your specific hardware
- **Minimal time-to-first-token** (TTFT) for responsive applications
- **Resource utilization** tracking and optimization
- **Reproducible benchmarking** with detailed logging

### Why O3?

- **Hardware-First Approach:** Optimizes for your specific GPU/CPU configuration
- **Agentic Workflow Ready:** Maximizes context length for tool traces and memory
- **Production Ready:** Provides stable, tested configurations
- **Team Friendly:** Generates shareable logs and reports
- **Extensible:** Easy to add new models and test scenarios

## 🚀 Quick Start

### Installation

```bash
# Clone or download the O3 suite files
# Install dependencies
pip install -r requirements.txt

# Verify Ollama is running
ollama list
```

### Basic Usage

```bash
# Test a single model
python o3_optimizer.py qwen3-coder:30b

# Test multiple models
python o3_optimizer.py qwen3-coder:30b gemma3:latest

# Test all your models
python o3_optimizer.py qwen3-coder:30b orieg/gemma3-tools:27b-it-qat liquid-rag:latest qwen2.5:3b-instruct gemma3:latest

# Generate comprehensive report
python o3_report_generator.py --csv
```

### VS Code Integration

Use Ctrl+Shift+P → "Tasks: Run Task" → Select O3 task:
- **O3: Test Single Model** - Interactive model selection
- **O3: Test All Coding Models** - Optimize coding-focused models
- **O3: Full Test Suite** - Test all supported models
- **O3: Generate Summary Report** - Create markdown report

## 📊 What O3 Tests

### Optimization Parameters

| Parameter | Purpose | Tested Values |
|-----------|---------|---------------|
| `num_ctx` | Context window size | 4096, 8192, 12288, 16384, 24576, 32768 |
| `batch` | Batch size for processing | 8, 16, 32 |
| `f16_kv` | KV cache precision | true, false |
| `num_predict` | Output token limit | 256, 512 |
| `num_thread` | CPU threads | Physical core count |

### Measured Metrics

- **Time to First Token (TTFT)** - Response latency
- **Tokens per Second** - Throughput
- **VRAM Usage** - GPU memory consumption  
- **RAM Usage** - System memory consumption
- **Stability** - Success rate across concurrent runs
- **Context Limits** - Maximum stable context window

## 🏗️ Architecture

```
O3 (Ozone) Test Suite
├── o3_optimizer.py          # Main test runner
├── o3_report_generator.py   # Report and analysis generator
├── requirements.txt         # Python dependencies
├── .vscode/
│   ├── tasks.json          # VS Code integration
│   └── launch.json         # Debug configurations
└── o3_results/             # Generated results
    ├── logs/               # Detailed JSONL logs
    ├── summaries/          # Per-model JSON summaries  
    ├── defaults/           # Recommended YAML configs
    └── env/                # System environment snapshots
```

## 📈 Model-Specific Configurations

O3 includes optimized test grids for common models:

### Large Coding Models (27B-30B)
- **Models:** `qwen3-coder:30b`, `orieg/gemma3-tools:27b-it-qat`
- **Focus:** Maximum context for complex code generation
- **Batch Sizes:** 8, 16 (VRAM conscious)
- **Context Range:** 4096 → 32768 tokens

### RAG Models  
- **Models:** `liquid-rag:latest`
- **Focus:** Fast retrieval and response generation
- **Batch Sizes:** 16, 32
- **Context Range:** 8192 → 32768 tokens

### Chat/Instruct Models (3B-8B)
- **Models:** `qwen2.5:3b-instruct`, `gemma3:latest`
- **Focus:** Balanced performance and context
- **Batch Sizes:** 16, 32
- **Context Range:** 4096 → 32768 tokens

## 🔧 Advanced Usage

### Custom Model Testing

```bash
# Test custom model with specific concurrency levels
python o3_optimizer.py my-custom-model:latest --concurrency 1 2 4

# Save to custom directory
python o3_optimizer.py model-name --output-dir custom_results
```

### Programmatic Usage

```python
from o3_optimizer import OllamaOptimizer

optimizer = OllamaOptimizer("results")
results = optimizer.test_model("qwen3-coder:30b")
optimizer.save_results("qwen3-coder:30b", results)
```

### Configuration Customization

Edit the `generate_test_configs()` method in `o3_optimizer.py` to:
- Add new models with custom parameter grids
- Modify context window ranges  
- Adjust batch size options
- Change test repetition counts

## 📊 Understanding Results

### Output Files

#### `defaults/model_name.yaml`
```yaml
model: qwen3-coder:30b
presets:
  max_ctx:
    num_ctx: 16384      # Maximum stable context
    batch: 16
    f16_kv: true
    tokens_per_sec: 12.5
  fast_ctx:  
    num_ctx: 8192       # Optimized for speed
    batch: 16
    f16_kv: true
    tokens_per_sec: 18.3
```

#### `summaries/model_name.json`
```json
{
  "model": "qwen3-coder:30b",
  "total_tests": 48,
  "successful_tests": 42,
  "max_ctx_preset": { ... },
  "fast_ctx_preset": { ... }
}
```

#### `logs/model_name.jsonl`
```json
{"timestamp": "...", "model": "...", "config": {...}, "ttft_ms": 850, "tokens_per_sec": 12.5, ...}
{"timestamp": "...", "model": "...", "config": {...}, "ttft_ms": 1200, "tokens_per_sec": 8.3, ...}
```

### Performance Interpretation

**Tokens/Second:**
- **>20 tok/s** - Excellent for real-time applications
- **10-20 tok/s** - Good for most interactive use cases  
- **5-10 tok/s** - Acceptable for batch processing
- **<5 tok/s** - Consider reducing context or using smaller model

**Time to First Token:**
- **<500ms** - Excellent responsiveness
- **500-1000ms** - Good for interactive use
- **1-2s** - Acceptable for most applications
- **>2s** - Consider optimizing for faster preset

## 🎯 Integration Examples

### Use with Ollama API

```python
import requests

# Use O3 optimized settings
response = requests.post('http://localhost:11434/api/generate', json={
    "model": "qwen3-coder:30b",
    "prompt": "Generate a Python class for...",
    "options": {
        "num_ctx": 16384,    # From O3 max_ctx preset
        "batch": 16,
        "f16_kv": True,
        "num_predict": 512,
        "temperature": 0.2,
        "top_p": 0.95
    }
})
```

### Use in LangChain

```python
from langchain_community.llms import Ollama

# Initialize with O3 optimized parameters
llm = Ollama(
    model="qwen3-coder:30b",
    num_ctx=16384,          # Max context from O3
    batch_size=16,
    f16_kv=True,
    temperature=0.2
)
```

### Use in Agentic Frameworks

O3 settings are particularly valuable for:
- **AutoGPT/AgentGPT** - Long conversation history
- **LangGraph** - Multi-step tool usage  
- **CrewAI** - Complex multi-agent interactions
- **Custom RAG** - Large document contexts

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory" / "ROCm out of memory"**
```bash
# Use smaller batch sizes
python o3_optimizer.py model-name  # Will test smaller batches first
```

**Slow performance**
```bash
# Check system resources
nvidia-smi  # or rocm-smi
htop
```

**Model not found**
```bash
# Pull model first
ollama pull qwen3-coder:30b
ollama list
```

### Performance Tuning

1. **Thermal Throttling:** Monitor GPU temperature during extended tests
2. **Memory Pressure:** Close other applications before testing
3. **Driver Issues:** Update GPU drivers if experiencing crashes
4. **Concurrent Usage:** Avoid running other Ollama instances during tests

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: O3 Model Optimization

on: 
  schedule:
    - cron: '0 2 * * 0'  # Weekly optimization runs

jobs:
  optimize:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run O3 optimization  
        run: python o3_optimizer.py qwen3-coder:30b
      - name: Generate report
        run: python o3_report_generator.py --csv
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: o3-results
          path: o3_results/
```

## 🤝 Contributing

O3 is designed to be extended and customized:

1. **Add New Models:** Update `generate_test_configs()` with model-specific parameters
2. **Add Metrics:** Extend `TestResult` class and monitoring
3. **Add Test Types:** Create specialized test scenarios
4. **Improve Reporting:** Enhance markdown and CSV output formats

## 📜 License

MIT License - Feel free to use in educational and commercial projects.

## 🙋 Support

- **Issues:** Hardware-specific optimization problems
- **Features:** New model support, additional metrics
- **Integration:** Help with agentic frameworks and production deployment

---

**Built for educators, developers, and AI practitioners who need reliable, optimized model performance.**

*Optimize once, deploy confidently.*
'''

with open("README.md", "w") as f:
    f.write(readme_content)

print("Created README.md")

# Create quick start script
quickstart_code = '''#!/usr/bin/env python3
"""
O3 Quick Start - Automated setup and initial test
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version}")
    
    # Check if Ollama is installed and running
    if not run_command("ollama --version", "Check Ollama installation"):
        print("❌ Ollama not found. Please install from https://ollama.ai")
        return False
    
    if not run_command("ollama list", "Check Ollama service"):
        print("❌ Ollama service not running. Please start with 'ollama serve'")
        return False
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Install Python dependencies")

def detect_models():
    """Detect available models"""
    try:
        result = subprocess.run("ollama list", shell=True, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\\n')[1:]  # Skip header
        
        models = []
        target_models = [
            "qwen3-coder:30b", "orieg/gemma3-tools:27b-it-qat", 
            "liquid-rag:latest", "qwen2.5:3b-instruct", "gemma3:latest"
        ]
        
        for line in lines:
            if line.strip():
                model_name = line.split()[0]
                if model_name in target_models:
                    models.append(model_name)
        
        return models
    except subprocess.CalledProcessError:
        return []

def run_sample_test(models):
    """Run a sample optimization test"""
    if not models:
        print("❌ No supported models found")
        return False
    
    # Use smallest available model for quick test
    test_model = models[0]
    print(f"\\n🧪 Running sample test with {test_model}...")
    
    return run_command(f"python o3_optimizer.py {test_model} --concurrency 1", 
                      f"Sample optimization test")

def generate_sample_report():
    """Generate a sample report"""
    return run_command("python o3_report_generator.py --csv", "Generate sample report")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   O3 (Ozone) Quick Start                    ║
║              Ollama Open-Source Optimizer                   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\\n❌ Prerequisites check failed. Please resolve issues and try again.")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\\n❌ Failed to install dependencies.")
        return 1
    
    # Detect models
    print("\\n🔍 Detecting available models...")
    available_models = detect_models()
    
    if available_models:
        print(f"✅ Found {len(available_models)} supported models:")
        for model in available_models:
            print(f"   - {model}")
    else:
        print("⚠️  No supported models found. You may need to pull models first:")
        print("   ollama pull qwen2.5:3b-instruct")
        print("   ollama pull gemma3:latest")
    
    # Offer to run sample test
    if available_models:
        response = input(f"\\n❓ Run sample optimization test with {available_models[0]}? (y/N): ").lower()
        if response in ['y', 'yes']:
            if run_sample_test(available_models):
                print("\\n🎉 Sample test completed successfully!")
                
                # Generate report
                if generate_sample_report():
                    print("\\n📊 Sample report generated!")
                    
                    # Show next steps
                    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                        Next Steps                            ║
╠══════════════════════════════════════════════════════════════╣
║ 1. Check results in 'o3_results/' directory                 ║
║ 2. View the generated report: O3_Report_*.md                ║
║ 3. Check optimized settings: o3_results/defaults/           ║
║ 4. Run full test suite:                                     ║
║    python o3_optimizer.py {' '.join(available_models[:3])}   ║
║                                                              ║
║ 🚀 VS Code users: Use Ctrl+Shift+P → "Tasks: Run Task"     ║
╚══════════════════════════════════════════════════════════════╝
                    """)
                else:
                    print("⚠️  Report generation failed, but test data is available")
            else:
                print("❌ Sample test failed")
        else:
            print("\\nSetup complete! Run 'python o3_optimizer.py <model-name>' to start optimizing.")
    
    print("\\n✅ O3 Quick Start completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

with open("quickstart.py", "w") as f:
    f.write(quickstart_code)

print("Created quickstart.py")

# Create example configuration file
example_config = '''# Example O3 Configuration
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
'''

with open("o3_config.py", "w") as f:
    f.write(example_config)

print("Created o3_config.py (example configuration)")

print("\\n🎉 O3 (Ozone) Test Suite - All Materials Created!")
print("\\nFiles created:")
print("- o3_optimizer.py (main test runner)")
print("- o3_report_generator.py (report generator)")  
print("- requirements.txt (Python dependencies)")
print("- quickstart.py (automated setup)")
print("- README.md (comprehensive documentation)")
print("- o3_config.py (example configuration)")
print("- .vscode/tasks.json (VS Code integration)")
print("- .vscode/launch.json (debugging configuration)")
print("\\nTo get started:")
print("1. Run: python quickstart.py")
print("2. Or manually: pip install -r requirements.txt")
print("3. Then: python o3_optimizer.py <model-name>")