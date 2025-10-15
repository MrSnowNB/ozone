# O3 (Ozone) - Tech Team Handoff Document

## Project Overview
**O3 (Ollama Open-Source Optimizer)** is a hardware-focused performance optimization suite designed to find optimal settings for Ollama models on specific hardware configurations. Built for Stage 1: discovering optimal context lengths and performance settings for agentic workflows.

## Business Context
- **Primary Goal:** Maximize context window size for agentic AI workflows
- **Secondary Goal:** Optimize throughput (tokens/second) and minimize latency
- **Target Environment:** HP Strix Halo workstations, AMD GPUs, 64GB RAM/VRAM
- **Use Case:** Educational AI systems, RAG applications, coding assistance

## Technical Architecture

### Core Components
1. **o3_optimizer.py** - Main test runner with hardware monitoring
2. **o3_report_generator.py** - Comprehensive reporting and analysis
3. **VS Code Integration** - Tasks and debugging configuration
4. **Automated Setup** - quickstart.py for streamlined deployment

### Test Strategy
- **Parameter Grid Search:** Context size, batch size, precision settings
- **Hardware Monitoring:** VRAM/RAM usage, performance metrics
- **Stability Testing:** Concurrent load testing, OOM detection
- **Reproducible Logging:** JSONL format with full environment capture

### Models Tested
- `qwen3-coder:30b` - Large coding model (18GB)
- `orieg/gemma3-tools:27b-it-qat` - Quantized tools model (18GB)
- `liquid-rag:latest` - RAG-optimized model (730MB)
- `qwen2.5:3b-instruct` - Small instruct model (1.9GB)
- `gemma3:latest` - General purpose model (3.3GB)

## Implementation Details

### Key Optimizations Tested
```yaml
Parameters:
  num_ctx: [4096, 8192, 12288, 16384, 24576, 32768]  # Context window
  batch: [8, 16, 32]                                 # Processing batch size
  f16_kv: [true, false]                             # KV cache precision
  num_predict: [256, 512]                           # Output limit
  num_thread: <physical_cores>                      # CPU threading
```

### Performance Metrics
- **Time to First Token (TTFT)** - Response latency
- **Tokens per Second** - Sustained throughput
- **Memory Usage** - Peak VRAM/RAM consumption
- **Stability Score** - Success rate under concurrent load
- **Context Ceiling** - Maximum stable context window

### Output Format
```
o3_results/
├── logs/           # Detailed JSONL test logs
├── summaries/      # Per-model performance summaries
├── defaults/       # Recommended YAML configurations  
└── env/            # System environment snapshots
```

## Deployment Instructions

### Prerequisites
- Python 3.7+
- Ollama installed and running
- Target models pulled locally
- GPU monitoring tools (nvidia-smi/rocm-smi)

### Quick Deployment
```bash
# Automated setup
python quickstart.py

# Manual setup  
pip install -r requirements.txt
python o3_optimizer.py qwen3-coder:30b
python o3_report_generator.py --csv
```

### VS Code Integration
- Import workspace settings from .vscode/
- Use Ctrl+Shift+P → "Tasks: Run Task" → Select O3 operation
- Debug configurations included for development

## Expected Outcomes

### Deliverables
1. **Optimal Settings** - Max context and fast presets per model
2. **Performance Baselines** - Throughput and latency benchmarks  
3. **Resource Requirements** - VRAM/RAM usage profiles
4. **Stability Thresholds** - Safe operating parameters
5. **Comprehensive Reports** - Markdown and CSV formats

### Success Criteria
- Identify maximum stable context window per model
- Achieve >10 tokens/sec for 30B models, >20 tokens/sec for smaller models
- <2 second TTFT for interactive applications
- <90% VRAM utilization for stability margin
- Zero OOM failures in recommended configurations

## Production Considerations

### Monitoring Requirements
- GPU temperature and clock speed monitoring
- Memory usage alerts at 85% utilization
- Performance regression detection
- Concurrent load testing validation

### Scaling Considerations  
- Results specific to tested hardware configuration
- Re-optimization required for different GPU/RAM configurations
- Model-specific settings may need adjustment for fine-tuned variants
- Batch processing may require different optimization profiles

### Risk Mitigation
- All tests include OOM protection and timeouts
- Progressive context testing prevents system crashes  
- Full environment logging enables issue reproduction
- Rollback configurations included in defaults

## Integration Points

### API Usage
```python
# Optimized Ollama API calls
requests.post('http://localhost:11434/api/generate', json={
    "model": "qwen3-coder:30b",
    "options": {
        "num_ctx": 16384,      # From O3 max_ctx preset
        "batch": 16,
        "f16_kv": True
    }
})
```

### Framework Integration
- LangChain: Use optimized context windows
- AutoGPT: Apply fast presets for tool calls
- RAG Systems: Use max context for document processing
- Educational Platforms: Stable settings for classroom use

## Next Phase Planning

### Stage 2 Expansion
- Automated hyperparameter tuning
- Multi-GPU load balancing
- Quality metrics integration
- Dashboard web interface

### Operational Integration
- CI/CD pipeline integration
- Automated regression testing
- Performance monitoring alerts
- Configuration management system

## Support Resources

### Documentation
- README.md - Complete user guide
- Code comments - Technical implementation details
- Example configurations - Common use case templates

### Troubleshooting
- Hardware compatibility matrix
- Common error resolution guide
- Performance tuning recommendations
- Contact information for technical support

---

**Project Status:** Ready for deployment and Stage 1 testing
**Next Review:** After initial optimization runs complete
**Stakeholders:** Education team, infrastructure team, AI development team
