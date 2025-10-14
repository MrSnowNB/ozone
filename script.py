# Create the main Python test runner for O3 (Ollama Open-Source Optimizer)
test_runner_code = '''#!/usr/bin/env python3
"""
Ollama Open-Source Optimizer (O3/Ozone) - Hardware Performance Test Suite
Stage 1: Context Window and Performance Optimization

Focuses on finding optimal hardware settings for maximum context length
and throughput, designed for agentic workflows.
"""

import subprocess
import json
import yaml
import time
import datetime
import os
import sys
import psutil
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import threading

@dataclass
class TestConfig:
    """Configuration for a single test run"""
    model: str
    num_ctx: int
    batch: int
    num_predict: int
    num_thread: int
    f16_kv: bool
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 42

@dataclass
class TestResult:
    """Results from a single test run"""
    timestamp: str
    run_id: str
    model: str
    model_digest: str
    config: TestConfig
    success: bool
    error: Optional[str]
    ttft_ms: Optional[float]
    total_ms: Optional[float]
    output_tokens: Optional[int]
    tokens_per_sec: Optional[float]
    vram_before_mb: Optional[int]
    vram_after_mb: Optional[int]
    ram_before_mb: Optional[int]
    ram_after_mb: Optional[int]
    concurrency_level: int
    run_index: int

class HardwareMonitor:
    """Monitor system resources during tests"""
    
    def __init__(self):
        self.gpu_type = self._detect_gpu()
    
    def _detect_gpu(self) -> str:
        """Detect GPU type (AMD/NVIDIA/None)"""
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            return "nvidia"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        try:
            subprocess.run(["rocm-smi"], capture_output=True, check=True)
            return "amd"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return "none"
    
    def get_vram_usage(self) -> Optional[int]:
        """Get current VRAM usage in MB"""
        if self.gpu_type == "nvidia":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True
                )
                return int(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError):
                return None
        
        elif self.gpu_type == "amd":
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmemuse", "--csv"],
                    capture_output=True, text=True, check=True
                )
                # Parse AMD output - this is approximate
                lines = result.stdout.strip().split('\\n')
                if len(lines) > 1:
                    # Extract memory usage from CSV format
                    data = lines[1].split(',')
                    if len(data) > 3:
                        return int(float(data[3]) * 1024)  # Convert GB to MB
            except (subprocess.CalledProcessError, ValueError, IndexError):
                return None
        
        return None
    
    def get_ram_usage(self) -> int:
        """Get current RAM usage in MB"""
        return int(psutil.virtual_memory().used / 1024 / 1024)

class OllamaOptimizer:
    """Main optimizer class for O3 test suite"""
    
    def __init__(self, output_dir: str = "o3_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.monitor = HardwareMonitor()
        
        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        (self.output_dir / "defaults").mkdir(exist_ok=True)
        (self.output_dir / "env").mkdir(exist_ok=True)
        
        self.test_prompt = "def fibonacci(n: int) -> int:\\n    # Generate fibonacci sequence up to n terms"
        self.warmup_prompt = "Hello, this is a warmup test."
    
    def capture_environment(self) -> Dict:
        """Capture system environment information"""
        env_info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "os": os.uname()._asdict() if hasattr(os, 'uname') else str(os.name),
            "python_version": sys.version,
            "cpu_info": {
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_ram_gb": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
                "available_ram_gb": round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 2)
            },
            "gpu_type": self.monitor.gpu_type
        }
        
        # Get Ollama version
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            env_info["ollama_version"] = result.stdout.strip()
        except subprocess.CalledProcessError:
            env_info["ollama_version"] = "unknown"
        
        # Get model list
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            env_info["available_models"] = result.stdout
        except subprocess.CalledProcessError:
            env_info["available_models"] = "unknown"
        
        return env_info
    
    def get_model_digest(self, model: str) -> str:
        """Get model digest/hash"""
        try:
            result = subprocess.run(
                ["ollama", "show", model, "--modelfile"], 
                capture_output=True, text=True, check=True
            )
            # Extract digest from modelfile output
            for line in result.stdout.split('\\n'):
                if line.startswith('FROM'):
                    parts = line.split('@')
                    if len(parts) > 1:
                        return parts[1][:12]  # First 12 chars of digest
            return "unknown"
        except subprocess.CalledProcessError:
            return "unknown"
    
    def warmup_model(self, model: str) -> bool:
        """Warmup model with a simple prompt"""
        print(f"  Warming up {model}...")
        try:
            subprocess.run([
                "ollama", "run", model,
                "--num-ctx", "2048",
                "--num-predict", "10",
                self.warmup_prompt
            ], capture_output=True, timeout=30, check=True)
            time.sleep(2)  # Cool down
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    def run_single_test(self, config: TestConfig, run_id: str, 
                       concurrency_level: int, run_index: int) -> TestResult:
        """Run a single test with given configuration"""
        
        # Get model digest
        model_digest = self.get_model_digest(config.model)
        
        # Capture before state
        vram_before = self.monitor.get_vram_usage()
        ram_before = self.monitor.get_ram_usage()
        
        # Build ollama command
        cmd = [
            "ollama", "run", config.model,
            "--num-ctx", str(config.num_ctx),
            "--batch", str(config.batch),
            "--num-predict", str(config.num_predict),
            "--num-thread", str(config.num_thread),
            "--temperature", str(config.temperature),
            "--top-p", str(config.top_p),
            "--seed", str(config.seed)
        ]
        
        if config.f16_kv:
            cmd.extend(["--f16-kv", "true"])
        
        cmd.append(self.test_prompt)
        
        start_time = time.time()
        ttft_time = None
        
        try:
            # Run with timeout
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            # Monitor for first token (approximate)
            first_output = False
            output_lines = []
            
            while True:
                line = process.stdout.readline()
                if line:
                    if not first_output:
                        ttft_time = time.time() - start_time
                        first_output = True
                    output_lines.append(line)
                elif process.poll() is not None:
                    break
            
            total_time = time.time() - start_time
            return_code = process.wait(timeout=90)
            
            if return_code != 0:
                stderr_output = process.stderr.read()
                raise subprocess.CalledProcessError(return_code, cmd, stderr_output)
            
            # Parse output for token count (approximate)
            full_output = ''.join(output_lines)
            output_tokens = len(full_output.split()) if full_output else 0
            tokens_per_sec = output_tokens / total_time if total_time > 0 else 0
            
            # Capture after state
            time.sleep(1)  # Let system stabilize
            vram_after = self.monitor.get_vram_usage()
            ram_after = self.monitor.get_ram_usage()
            
            return TestResult(
                timestamp=datetime.datetime.now().isoformat(),
                run_id=run_id,
                model=config.model,
                model_digest=model_digest,
                config=config,
                success=True,
                error=None,
                ttft_ms=ttft_time * 1000 if ttft_time else None,
                total_ms=total_time * 1000,
                output_tokens=output_tokens,
                tokens_per_sec=tokens_per_sec,
                vram_before_mb=vram_before,
                vram_after_mb=vram_after,
                ram_before_mb=ram_before,
                ram_after_mb=ram_after,
                concurrency_level=concurrency_level,
                run_index=run_index
            )
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return TestResult(
                timestamp=datetime.datetime.now().isoformat(),
                run_id=run_id,
                model=config.model,
                model_digest=model_digest,
                config=config,
                success=False,
                error=str(e),
                ttft_ms=None,
                total_ms=None,
                output_tokens=None,
                tokens_per_sec=None,
                vram_before_mb=vram_before,
                vram_after_mb=None,
                ram_before_mb=ram_before,
                ram_after_mb=None,
                concurrency_level=concurrency_level,
                run_index=run_index
            )
    
    def generate_test_configs(self, model: str) -> List[TestConfig]:
        """Generate test configurations for a model"""
        
        # Model-specific parameter grids
        model_configs = {
            "qwen3-coder:30b": {
                "num_ctx": [4096, 8192, 12288, 16384, 24576, 32768],
                "batch": [8, 16],
                "f16_kv": [True, False],
                "num_predict": [256, 512]
            },
            "orieg/gemma3-tools:27b-it-qat": {
                "num_ctx": [4096, 8192, 12288, 16384, 24576],
                "batch": [8, 16],
                "f16_kv": [True, False],
                "num_predict": [256, 512]
            },
            "liquid-rag:latest": {
                "num_ctx": [8192, 16384, 24576, 32768],
                "batch": [16, 32],
                "f16_kv": [True],
                "num_predict": [256, 512]
            },
            "qwen2.5:3b-instruct": {
                "num_ctx": [8192, 16384, 24576, 32768],
                "batch": [16, 32],
                "f16_kv": [True],
                "num_predict": [256, 512]
            },
            "gemma3:latest": {
                "num_ctx": [4096, 8192, 12288, 16384],
                "batch": [16, 32],
                "f16_kv": [True],
                "num_predict": [256, 512]
            }
        }
        
        # Default config for unknown models
        default_config = {
            "num_ctx": [4096, 8192, 16384],
            "batch": [16],
            "f16_kv": [True],
            "num_predict": [256]
        }
        
        config_params = model_configs.get(model, default_config)
        num_thread = psutil.cpu_count(logical=False) or 8  # Physical cores
        
        configs = []
        
        # Start with smaller contexts and work up
        for num_ctx in sorted(config_params["num_ctx"]):
            for batch in config_params["batch"]:
                for f16_kv in config_params["f16_kv"]:
                    for num_predict in config_params["num_predict"]:
                        configs.append(TestConfig(
                            model=model,
                            num_ctx=num_ctx,
                            batch=batch,
                            num_predict=num_predict,
                            num_thread=num_thread,
                            f16_kv=f16_kv
                        ))
        
        return configs
    
    def test_model(self, model: str, concurrency_levels: List[int] = [1, 2]) -> List[TestResult]:
        """Test a single model with all configurations"""
        print(f"\\n=== Testing {model} ===")
        
        # Warmup
        if not self.warmup_model(model):
            print(f"Failed to warmup {model}, skipping...")
            return []
        
        configs = self.generate_test_configs(model)
        all_results = []
        
        for config in configs:
            print(f"Testing ctx={config.num_ctx}, batch={config.batch}, f16_kv={config.f16_kv}, predict={config.num_predict}")
            
            for concurrency in concurrency_levels:
                print(f"  Concurrency level: {concurrency}")
                
                if concurrency == 1:
                    # Single run
                    for run_idx in range(3):  # 3 repetitions
                        run_id = f"{model}_{config.num_ctx}_{config.batch}_{concurrency}_{run_idx}"
                        result = self.run_single_test(config, run_id, concurrency, run_idx)
                        all_results.append(result)
                        
                        if not result.success:
                            print(f"    Run {run_idx + 1}: FAILED - {result.error}")
                            # Skip remaining configs if we hit OOM
                            if "memory" in str(result.error).lower():
                                print(f"    Memory error detected, stopping context expansion")
                                return all_results
                        else:
                            print(f"    Run {run_idx + 1}: {result.tokens_per_sec:.1f} tok/s, TTFT: {result.ttft_ms:.0f}ms")
                
                else:
                    # Concurrent runs
                    def run_concurrent(run_idx):
                        run_id = f"{model}_{config.num_ctx}_{config.batch}_{concurrency}_{run_idx}"
                        return self.run_single_test(config, run_id, concurrency, run_idx)
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [executor.submit(run_concurrent, i) for i in range(concurrency)]
                        concurrent_results = [future.result() for future in futures]
                        all_results.extend(concurrent_results)
                        
                        successful = [r for r in concurrent_results if r.success]
                        if successful:
                            avg_tps = sum(r.tokens_per_sec for r in successful) / len(successful)
                            avg_ttft = sum(r.ttft_ms for r in successful) / len(successful)
                            print(f"    Concurrent avg: {avg_tps:.1f} tok/s, TTFT: {avg_ttft:.0f}ms")
                        else:
                            print(f"    Concurrent: ALL FAILED")
                            return all_results
        
        return all_results
    
    def save_results(self, model: str, results: List[TestResult]):
        """Save test results to files"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSONL log
        log_file = self.output_dir / "logs" / f"{model.replace(':', '_')}_{timestamp}.jsonl"
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'w') as f:
            for result in results:
                json.dump(asdict(result), f)
                f.write('\\n')
        
        # Generate summary
        successful_results = [r for r in results if r.success]
        if not successful_results:
            return
        
        # Find best configurations
        max_ctx_result = max(successful_results, key=lambda r: r.config.num_ctx)
        fast_results = [r for r in successful_results if r.concurrency_level == 1]
        if fast_results:
            fast_ctx_result = max(fast_results, key=lambda r: r.tokens_per_sec)
        else:
            fast_ctx_result = max_ctx_result
        
        summary = {
            "model": model,
            "timestamp": timestamp,
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "max_ctx_preset": {
                "num_ctx": max_ctx_result.config.num_ctx,
                "batch": max_ctx_result.config.batch,
                "f16_kv": max_ctx_result.config.f16_kv,
                "num_predict": max_ctx_result.config.num_predict,
                "tokens_per_sec": max_ctx_result.tokens_per_sec,
                "ttft_ms": max_ctx_result.ttft_ms
            },
            "fast_ctx_preset": {
                "num_ctx": fast_ctx_result.config.num_ctx,
                "batch": fast_ctx_result.config.batch,
                "f16_kv": fast_ctx_result.config.f16_kv,
                "num_predict": fast_ctx_result.config.num_predict,
                "tokens_per_sec": fast_ctx_result.tokens_per_sec,
                "ttft_ms": fast_ctx_result.ttft_ms
            }
        }
        
        # Save summary
        summary_file = self.output_dir / "summaries" / f"{model.replace(':', '_')}_{timestamp}.json"
        summary_file.parent.mkdir(exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update defaults
        defaults_file = self.output_dir / "defaults" / f"{model.replace(':', '_')}.yaml"
        defaults_file.parent.mkdir(exist_ok=True)
        
        defaults = {
            "model": model,
            "updated_at": datetime.datetime.now().isoformat(),
            "presets": {
                "max_ctx": {
                    "num_ctx": max_ctx_result.config.num_ctx,
                    "batch": max_ctx_result.config.batch,
                    "f16_kv": max_ctx_result.config.f16_kv,
                    "num_predict": max_ctx_result.config.num_predict,
                    "num_thread": max_ctx_result.config.num_thread,
                    "temperature": max_ctx_result.config.temperature,
                    "top_p": max_ctx_result.config.top_p,
                    "notes": f"Max stable context: {max_ctx_result.tokens_per_sec:.1f} tok/s"
                },
                "fast_ctx": {
                    "num_ctx": fast_ctx_result.config.num_ctx,
                    "batch": fast_ctx_result.config.batch,
                    "f16_kv": fast_ctx_result.config.f16_kv,
                    "num_predict": fast_ctx_result.config.num_predict,
                    "num_thread": fast_ctx_result.config.num_thread,
                    "temperature": fast_ctx_result.config.temperature,
                    "top_p": fast_ctx_result.config.top_p,
                    "notes": f"Optimized speed: {fast_ctx_result.tokens_per_sec:.1f} tok/s"
                }
            }
        }
        
        with open(defaults_file, 'w') as f:
            yaml.dump(defaults, f, default_flow_style=False)
        
        print(f"\\nResults saved:")
        print(f"  Log: {log_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Defaults: {defaults_file}")
        print(f"\\nRecommended settings for {model}:")
        print(f"  Max Context: {max_ctx_result.config.num_ctx} tokens ({max_ctx_result.tokens_per_sec:.1f} tok/s)")
        print(f"  Fast Context: {fast_ctx_result.config.num_ctx} tokens ({fast_ctx_result.tokens_per_sec:.1f} tok/s)")


def main():
    parser = argparse.ArgumentParser(description="O3 (Ozone) - Ollama Hardware Optimizer")
    parser.add_argument("models", nargs="+", help="Models to test")
    parser.add_argument("--output-dir", default="o3_results", help="Output directory")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2], 
                       help="Concurrency levels to test")
    
    args = parser.parse_args()
    
    optimizer = OllamaOptimizer(args.output_dir)
    
    # Capture environment
    env_info = optimizer.capture_environment()
    env_file = optimizer.output_dir / "env" / f"env_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    env_file.parent.mkdir(exist_ok=True)
    
    with open(env_file, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print(f"O3 (Ozone) Ollama Hardware Optimizer")
    print(f"Output directory: {optimizer.output_dir}")
    print(f"GPU detected: {optimizer.monitor.gpu_type}")
    print(f"Physical CPU cores: {env_info['cpu_info']['cores_physical']}")
    print(f"Total RAM: {env_info['memory']['total_ram_gb']} GB")
    
    # Test each model
    for model in args.models:
        try:
            results = optimizer.test_model(model, args.concurrency)
            if results:
                optimizer.save_results(model, results)
        except KeyboardInterrupt:
            print(f"\\nInterrupted during {model} testing")
            break
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue
    
    print(f"\\nO3 testing complete. Results in: {optimizer.output_dir}")


if __name__ == "__main__":
    main()
'''

# Save the test runner
with open("o3_optimizer.py", "w") as f:
    f.write(test_runner_code)

print("Created o3_optimizer.py - Main test runner script")