# Create the report generator script
report_generator_code = '''#!/usr/bin/env python3
"""
O3 Report Generator - Creates comprehensive reports from test results
"""

import json
import yaml
import os
import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class O3ReportGenerator:
    """Generate comprehensive reports from O3 test results"""
    
    def __init__(self, results_dir: str = "o3_results"):
        self.results_dir = Path(results_dir)
        self.output_file = f"O3_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    def load_all_results(self) -> Dict[str, Any]:
        """Load all test results and summaries"""
        data = {
            "summaries": {},
            "defaults": {},
            "environment": {},
            "raw_logs": {}
        }
        
        # Load summaries
        summaries_dir = self.results_dir / "summaries"
        if summaries_dir.exists():
            for summary_file in summaries_dir.glob("*.json"):
                with open(summary_file, 'r') as f:
                    model_name = summary_file.stem.split('_')[0]
                    data["summaries"][model_name] = json.load(f)
        
        # Load defaults
        defaults_dir = self.results_dir / "defaults"
        if defaults_dir.exists():
            for default_file in defaults_dir.glob("*.yaml"):
                with open(default_file, 'r') as f:
                    model_name = default_file.stem
                    data["defaults"][model_name] = yaml.safe_load(f)
        
        # Load latest environment
        env_dir = self.results_dir / "env"
        if env_dir.exists():
            env_files = list(env_dir.glob("*.json"))
            if env_files:
                latest_env = max(env_files, key=lambda f: f.stat().st_mtime)
                with open(latest_env, 'r') as f:
                    data["environment"] = json.load(f)
        
        # Load raw logs for detailed analysis
        logs_dir = self.results_dir / "logs"
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.jsonl"):
                model_name = log_file.stem.split('_')[0]
                logs = []
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
                data["raw_logs"][model_name] = logs
        
        return data
    
    def generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report"""
        
        report = f"""# O3 (Ozone) Ollama Hardware Optimization Report

**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report contains hardware optimization results for Ollama models tested on your system. The O3 (Ozone) test suite focused on finding optimal settings for maximum context length and throughput, specifically designed for agentic workflows.

## System Configuration

"""
        
        # Add system info
        if data["environment"]:
            env = data["environment"]
            report += f"""### Hardware
- **CPU:** {env['cpu_info']['cores_physical']} physical cores, {env['cpu_info']['cores_logical']} logical
- **RAM:** {env['memory']['total_ram_gb']} GB total, {env['memory']['available_ram_gb']} GB available
- **GPU:** {env['gpu_type']} detected
- **OS:** {env.get('os', 'Unknown')}

### Software
- **Ollama Version:** {env.get('ollama_version', 'Unknown')}
- **Python Version:** {env.get('python_version', 'Unknown')}

"""
        
        # Add model optimization results
        report += """## Model Optimization Results

"""
        
        if data["summaries"]:
            # Create comparison table
            report += """### Performance Summary

| Model | Max Context | Max Throughput | Fast Context | Fast Throughput | TTFT (ms) |
|-------|------------|----------------|--------------|-----------------|-----------|
"""
            
            for model_name, summary in data["summaries"].items():
                max_ctx = summary.get("max_ctx_preset", {})
                fast_ctx = summary.get("fast_ctx_preset", {})
                
                report += f"| {model_name} | {max_ctx.get('num_ctx', 'N/A')} | {max_ctx.get('tokens_per_sec', 0):.1f} tok/s | {fast_ctx.get('num_ctx', 'N/A')} | {fast_ctx.get('tokens_per_sec', 0):.1f} tok/s | {fast_ctx.get('ttft_ms', 0):.0f} |\\n"
            
            report += "\\n"
        
        # Add detailed results per model
        report += """### Detailed Model Results

"""
        
        for model_name, summary in data["summaries"].items():
            report += f"""#### {model_name}

**Test Results:**
- Total Tests: {summary.get('total_tests', 0)}
- Successful Tests: {summary.get('successful_tests', 0)}

**Recommended Settings:**

**Max Context Preset** (for maximum context length):
```yaml
num_ctx: {summary['max_ctx_preset']['num_ctx']}
batch: {summary['max_ctx_preset']['batch']}
f16_kv: {summary['max_ctx_preset']['f16_kv']}
num_predict: {summary['max_ctx_preset']['num_predict']}
```
- Performance: {summary['max_ctx_preset']['tokens_per_sec']:.1f} tokens/sec
- TTFT: {summary['max_ctx_preset']['ttft_ms']:.0f}ms

**Fast Context Preset** (for optimal speed):
```yaml
num_ctx: {summary['fast_ctx_preset']['num_ctx']}
batch: {summary['fast_ctx_preset']['batch']}
f16_kv: {summary['fast_ctx_preset']['f16_kv']}
num_predict: {summary['fast_ctx_preset']['num_predict']}
```
- Performance: {summary['fast_ctx_preset']['tokens_per_sec']:.1f} tokens/sec  
- TTFT: {summary['fast_ctx_preset']['ttft_ms']:.0f}ms

"""
        
        # Add implementation guide
        report += """## Implementation Guide

### Using Optimized Settings

To use these optimized settings with Ollama:

```bash
# Max context preset for qwen3-coder:30b
ollama run qwen3-coder:30b \\
  --num-ctx 16384 \\
  --batch 16 \\
  --f16-kv true \\
  --num-predict 512 \\
  "Your prompt here"

# Fast context preset for quick responses
ollama run qwen3-coder:30b \\
  --num-ctx 8192 \\
  --batch 16 \\
  --f16-kv true \\
  --num-predict 256 \\
  "Your prompt here"
```

### API Usage

For programmatic usage:

```python
import requests

# Using optimized settings via Ollama API
response = requests.post('http://localhost:11434/api/generate', json={
    "model": "qwen3-coder:30b",
    "prompt": "Your prompt here",
    "options": {
        "num_ctx": 16384,
        "batch": 16,
        "f16_kv": True,
        "num_predict": 512,
        "temperature": 0.2,
        "top_p": 0.95
    }
})
```

### Agentic Workflow Considerations

For agentic workflows, use the **Max Context** presets to:
- Maintain longer conversation history
- Include extensive tool traces
- Support complex multi-step reasoning
- Handle large document contexts

Use the **Fast Context** presets for:
- Quick tool responses
- Real-time interactions  
- High-frequency API calls
- Resource-constrained environments

"""
        
        # Add performance analysis
        if data["raw_logs"]:
            report += """## Performance Analysis

### Resource Utilization

"""
            
            for model_name, logs in data["raw_logs"].items():
                successful_logs = [log for log in logs if log['success']]
                if not successful_logs:
                    continue
                
                # Calculate average resource usage
                vram_usage = [log.get('vram_after_mb', 0) - log.get('vram_before_mb', 0) 
                             for log in successful_logs if log.get('vram_after_mb') and log.get('vram_before_mb')]
                ram_usage = [log.get('ram_after_mb', 0) - log.get('ram_before_mb', 0) 
                            for log in successful_logs if log.get('ram_after_mb') and log.get('ram_before_mb')]
                
                if vram_usage or ram_usage:
                    avg_vram = sum(vram_usage) / len(vram_usage) if vram_usage else 0
                    avg_ram = sum(ram_usage) / len(ram_usage) if ram_usage else 0
                    
                    report += f"""#### {model_name}
- Average VRAM Usage: {avg_vram:.0f} MB
- Average RAM Usage: {avg_ram:.0f} MB

"""
        
        # Add troubleshooting section
        report += """## Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce `num_ctx` to a lower value
- Set `f16_kv: false` to use quantized KV cache
- Reduce `batch` size
- Close other applications to free VRAM/RAM

**Slow Performance:**
- Use Fast Context presets for better throughput
- Increase `batch` size if VRAM allows
- Ensure `f16_kv: true` for better speed
- Check system temperature and thermal throttling

**Model Loading Issues:**
- Verify model is pulled: `ollama list`
- Check available disk space
- Restart Ollama service: `ollama serve`

### Performance Tuning Tips

1. **For Coding Tasks:** Use higher context with moderate batch sizes
2. **For RAG Tasks:** Optimize for fast TTFT with smaller batches  
3. **For Chat Tasks:** Balance context and speed based on conversation length
4. **For Production:** Always test with your specific hardware and workload

## Files Generated

This optimization run generated the following files:

- **Logs:** `o3_results/logs/` - Detailed test execution logs
- **Summaries:** `o3_results/summaries/` - Per-model performance summaries  
- **Defaults:** `o3_results/defaults/` - Recommended settings in YAML format
- **Environment:** `o3_results/env/` - System configuration snapshots

## Next Steps

1. **Validate Settings:** Test the recommended settings with your specific use cases
2. **Monitor Performance:** Use the provided configs in production and monitor for stability
3. **Iterate:** Re-run O3 after system changes or model updates
4. **Scale:** Apply learnings to other models and deployment configurations

---

*Report generated by O3 (Ozone) - Ollama Open-Source Optimizer*
"""
        
        return report
    
    def save_report(self, report_content: str):
        """Save the report to file"""
        with open(self.output_file, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to: {self.output_file}")
    
    def generate_csv_summary(self, data: Dict[str, Any]):
        """Generate CSV summary for spreadsheet analysis"""
        if not data["summaries"]:
            return
        
        # Prepare data for CSV
        csv_data = []
        for model_name, summary in data["summaries"].items():
            max_ctx = summary.get("max_ctx_preset", {})
            fast_ctx = summary.get("fast_ctx_preset", {})
            
            csv_data.append({
                "Model": model_name,
                "Max_Context": max_ctx.get('num_ctx', 0),
                "Max_Batch": max_ctx.get('batch', 0),
                "Max_F16_KV": max_ctx.get('f16_kv', False),
                "Max_Tokens_Per_Sec": max_ctx.get('tokens_per_sec', 0),
                "Max_TTFT_ms": max_ctx.get('ttft_ms', 0),
                "Fast_Context": fast_ctx.get('num_ctx', 0),
                "Fast_Batch": fast_ctx.get('batch', 0),
                "Fast_F16_KV": fast_ctx.get('f16_kv', False),
                "Fast_Tokens_Per_Sec": fast_ctx.get('tokens_per_sec', 0),
                "Fast_TTFT_ms": fast_ctx.get('ttft_ms', 0),
                "Total_Tests": summary.get('total_tests', 0),
                "Successful_Tests": summary.get('successful_tests', 0)
            })
        
        # Save CSV
        csv_file = f"O3_Summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            import pandas as pd
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False)
            print(f"CSV summary saved to: {csv_file}")
        except ImportError:
            # Fallback to manual CSV writing
            import csv
            with open(csv_file, 'w', newline='') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
            print(f"CSV summary saved to: {csv_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate O3 optimization reports")
    parser.add_argument("--results-dir", default="o3_results", help="Results directory")
    parser.add_argument("--csv", action="store_true", help="Also generate CSV summary")
    
    args = parser.parse_args()
    
    generator = O3ReportGenerator(args.results_dir)
    
    print("Loading O3 test results...")
    data = generator.load_all_results()
    
    if not data["summaries"]:
        print("No test results found. Run o3_optimizer.py first.")
        return
    
    print("Generating comprehensive report...")
    report = generator.generate_markdown_report(data)
    generator.save_report(report)
    
    if args.csv:
        print("Generating CSV summary...")
        generator.generate_csv_summary(data)
    
    print("Report generation complete!")


if __name__ == "__main__":
    main()
'''

# Save the report generator
with open("o3_report_generator.py", "w") as f:
    f.write(report_generator_code)

print("Created o3_report_generator.py")