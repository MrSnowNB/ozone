# O3 (Ozone) - Ollama Open-Source Optimizer

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

The AI First Open-Source Ollama Optimizer (O3 or Ozone) is fundamentally designed to automate complex AI workflows by prioritizing AI-first communication protocols. This approach ensures seamless, protocol-driven interactions among AI components, tools, and systems. At its core, O3 leverages standardized YAML configurations to define critical parameters such as safety thresholds, model optimization presets, and hardware resource boundaries. By abstracting these configurations into human-readable and machine-interpretable YAML files, O3 enables automated workflow orchestration, allowing developers and organizations to deploy AI systems with confidence in stability, performance, and scalability. This methodology transforms traditional setup processes into streamlined, reproducible automation, minimizing manual intervention and reducing error-prone configurations across diverse hardware environments.

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
