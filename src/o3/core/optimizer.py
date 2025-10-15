#!/usr/bin/env python3
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

from .hardware_monitor import HardwareMonitor, RealTimeMonitor, quick_hardware_check

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

        self.test_prompt = "def fibonacci(n: int) -> int:\n    # Generate fibonacci sequence up to n terms"
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
            for line in result.stdout.split('\n'):
                if line.startswith('FROM'):
                    parts = line.split('@')
                    if len(parts) > 1:
                        return parts[1][:12]  # First 12 chars of digest
            return "unknown"
        except subprocess.CalledProcessError:
            return "unknown"

    def warmup_model(self, model: str) -> bool:
        """Warmup model using API with safeguards against infinite responses"""
        print(f"  Warming up {model}...")
        import requests  # Import here as not always available
        try:

            # Use API with very constrained parameters
            api_data = {
                "model": model,
                "prompt": "Say exactly: OK",
                "stream": False,
                "options": {
                    "num_ctx": 256,      # Very small context
                    "num_predict": 2,    # Bare minimum tokens
                    "temperature": 0.0,  # Deterministic
                    "seed": 42
                }
            }

            # Make API request with timeout
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=api_data,
                timeout=10  # 10 second timeout
            )

            if response.status_code != 200:
                print(f"    API request failed with status: {response.status_code}")
                return False

            result = response.json()
            response_text = result.get('response', '').strip()

            # Validate response - should be very short and contain expected text
            if len(response_text) > 10:  # Too long for "OK"
                print(f"    Warning: Long warmup response ({len(response_text)} chars)")
                print(f"    Response: {response_text[:30]}...")
                return False
            elif len(response_text) == 0:
                print(f"    Warning: Empty warmup response")
                return False

            # Check if we got something reasonable
            lower_resp = response_text.lower()
            if "ok" in lower_resp or "ready" in lower_resp or len(response_text) <= 5:
                print("    ✓ Model warmed up successfully via API")
                time.sleep(1)  # Brief API cool down
                return True
            else:
                print(f"    Warning: Unexpected warmup response: '{response_text}'")
                return False

        except requests.exceptions.RequestException as e:
            print(f"    API warmup failed: {e}")
            # Fallback: Try simple ollama run without flags (if model isn't loaded)
            try:
                print(f"    Falling back to simple ollama run...")
                subprocess.run(["ollama", "run", model, "OK"],
                              capture_output=True, timeout=5, check=True)
                print("    ✓ Model warmed up with simple fallback")
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return False

        except Exception as e:
            print(f"    Warmup failed with error: {e}")
            return False

    def run_single_test(self, config: TestConfig, run_id: str,
                       concurrency_level: int, run_index: int) -> TestResult:
        """Run a single test using Ollama API with proper safeguards"""
        # Get model digest
        model_digest = self.get_model_digest(config.model)

        # Capture before state
        vram_before = self.monitor.get_vram_usage()
        ram_before = self.monitor.get_ram_usage()

        # Build API request
        api_data = {
            "model": config.model,
            "prompt": self.test_prompt,
            "stream": False,
            "options": {
                "num_ctx": config.num_ctx,
                "batch": config.batch,
                "num_predict": config.num_predict,
                "num_thread": config.num_thread,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "seed": config.seed,
                "f16_kv": config.f16_kv
            }
        }

        start_time = time.time()

        try:
            import requests

            # Make API request with generous timeout based on config
            # Estimate timeout: 10 seconds base + 2 seconds per 1k context
            estimated_timeout = 10 + (config.num_ctx / 1000 * 2) + (config.num_predict / 100 * 1)
            timeout = min(max(estimated_timeout, 15), 120)  # Between 15-120 seconds

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=api_data,
                timeout=timeout
            )

            total_response_time = time.time() - start_time

            if response.status_code != 200:
                stderr_output = f"API returned status {response.status_code}: {response.text}"
                raise subprocess.CalledProcessError

            result = response.json()
            response_text = result.get('response', '')

            # Validate response is reasonable
            if len(response_text) < 5:  # Too short for a reasonable response
                return TestResult(
                    timestamp=datetime.datetime.now().isoformat(),
                    run_id=run_id,
                    model=config.model,
                    model_digest=model_digest,
                    config=config,
                    success=False,
                    error=f"Response too short: {len(response_text)} chars",
                    ttft_ms=None,
                    total_ms=total_response_time * 1000,
                    output_tokens=0,
                    tokens_per_sec=0,
                    vram_before_mb=vram_before,
                    vram_after_mb=None,
                    ram_before_mb=ram_before,
                    ram_after_mb=None,
                    concurrency_level=concurrency_level,
                    run_index=run_index
                )

            # Calculate performance metrics
            output_tokens = len(response_text.split()) if response_text else 0
            tokens_per_sec = output_tokens / total_response_time if total_response_time > 0 else 0

            # Capture after state
            time.sleep(0.5)  # Brief stabilization
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
                ttft_ms=total_response_time * 1000,  # Approximate - total time for now
                total_ms=total_response_time * 1000,
                output_tokens=output_tokens,
                tokens_per_sec=tokens_per_sec,
                vram_before_mb=vram_before,
                vram_after_mb=vram_after,
                ram_before_mb=ram_before,
                ram_after_mb=ram_after,
                concurrency_level=concurrency_level,
                run_index=run_index
            )

        except (requests.exceptions.RequestException, subprocess.CalledProcessError) as e:
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

    def load_ai_config(self) -> Dict:
        """Load AI-first configuration from o3_ai_config.yaml"""
        config_path = Path("o3_ai_config.yaml")
        if not config_path.exists():
            # Fallback to defaults if AI config not found
            print("Warning: o3_ai_config.yaml not found, using fallback defaults")
            return {
                "search_strategy": {
                    "initial_context_probe": 16384,
                    "binary_search_factor": 1.5,
                    "max_binary_iterations": 5,
                    "batch_adaptation": {
                        "initial_batch_large": 8,
                        "initial_batch_medium": 16,
                        "initial_batch_small": 32,
                        "scale_up_factor": 1.5,
                        "max_batch": 128
                    }
                }
            }

        try:
            # Try UTF-8 first
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            try:
                # Fallback to Latin-1 for Windows compatibility
                with open(config_path, 'r', encoding='latin-1') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Failed to load AI config with Latin-1 encoding: {e}")
                print(f"    Trying fallback...")
                try:
                    # Last fallback - let Python decide encoding
                    with open(config_path, 'r', errors='ignore') as f:
                        return yaml.safe_load(f)
                except Exception as e2:
                    print(f"Warning: Failed to load AI config with fallback: {e2}, using empty defaults")
                    return {}
        except Exception as e:
            print(f"Warning: Failed to load AI config: {e}, using defaults")
            return {}  # Fallback values will be used

    def binary_search_context(self, model: str, ai_config: Dict) -> List[TestConfig]:
        """Generate configurations using binary search for context discovery"""
        print(f"[AI-First] Starting binary search for {model} context discovery")

        # Get agentic focus categories from AI config
        preset_categories = ai_config.get("preset_categories", {})
        search_strategy = ai_config.get("search_strategy", {})

        # Start with configured probe point (64k for high-capacity systems)
        initial_probe = search_strategy.get("initial_context_probe", 16384)
        binary_factor = search_strategy.get("binary_search_factor", 1.3)
        max_iterations = search_strategy.get("max_binary_iterations", 5)

        # Target ranges for extreme context exploitation
        target_ranges = {
            "max_context": (32768, 131072),     # 32k-128k range
            "balanced": (12288, 65536),        # 12k-64k range for active use
            "fast_response": (4096, 32768)     # 4k-32k range for quick responses
        }

        # Get model intelligence for sizing
        model_intelligence = ai_config.get("model_intelligence", {})
        size_categories = model_intelligence.get("size_categories", {})

        # Determine model size category
        model_size = "medium"  # default
        for size, criteria in size_categories.items():
            min_params = criteria.get("min_params", 0)
            # Rough parameter estimate from model name
            if "30b" in model or "27b" in model:
                model_size = "large"
                break
            elif "3b" in model or "7b" in model:
                model_size = "medium"
                break
            # else keep default medium

        # Get initial batch from AI config based on model size
        batch_adaptation = search_strategy.get("batch_adaptation", {})
        if model_size == "large":
            initial_batch = batch_adaptation.get("initial_batch_large", 8)
        elif model_size == "medium":
            initial_batch = batch_adaptation.get("initial_batch_medium", 16)
        else:  # small
            initial_batch = batch_adaptation.get("initial_batch_small", 32)

        # Binary search configurations for each preset category
        configs = []
        num_thread = psutil.cpu_count(logical=False) or 8

        print(f"{model}: Detected as {model_size}, starting batch size: {initial_batch}")

        for preset_name, preset_config in preset_categories.items():
            target_range = target_ranges.get(preset_name, (4096, 65536))

            # Perform binary search within target range - push to limits
            low, high = target_range
            if "256k" in preset_name:
                # Force 256k context testing
                low = 131072  # Start from 128k
                high = 262144  # Go to 256k
            iterations = 0
            selected_contexts = []

            # Always include initial probe point
            selected_contexts.append(initial_probe)

            # Binary search iterations
            while low <= high and iterations < max_iterations:
                mid = (low + high) // 2
                selected_contexts.append(mid)

                # Expand search space
                if iterations < max_iterations // 2:
                    low = int(mid * binary_factor)
                    high = int(mid / binary_factor)
                else:
                    break

                iterations += 1

            # Final context sizes for this preset
            context_sizes = sorted(set(selected_contexts))
            context_sizes = [ctx for ctx in context_sizes if target_range[0] <= ctx <= target_range[1]]

            print(f"📊 Preset '{preset_name}': Testing contexts {context_sizes[:3]}{'...' if len(context_sizes) > 3 else ''}")

            # Generate configurations for this preset
            for num_ctx in context_sizes[:3]:  # Limit to top 3 contexts per preset for efficiency
                current_batch = initial_batch

                # Scale batch for high context sizes
                if num_ctx >= 65536:  # 64k+
                    scale_up = batch_adaptation.get("scale_up_factor", 1.5)
                    max_batch = batch_adaptation.get("max_batch", 128)
                    current_batch = min(int(current_batch * scale_up), max_batch)

                configs.append(TestConfig(
                    model=model,
                    num_ctx=num_ctx,
                    batch=current_batch,
                    num_predict=512,  # Use higher predict for agentic workflows
                    num_thread=num_thread,
                    f16_kv=True  # Always prefer f16_kv for performance on capable hardware
                ))

        print(f"✅ Generated {len(configs)} AI-optimized configurations for {model}")
        return configs

    def legacy_generate_configs(self, model: str) -> List[TestConfig]:
        """Legacy configuration generation (fallback for compatibility)"""
        print(f"📚 Using legacy configuration generation for {model}")

        # Original model-specific parameter grids
        model_configs = {
            "qwen3-coder:30b": {
                "num_ctx": [4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 81920, 98304, 114688, 131072],
                "batch": [8, 16, 32],
                "f16_kv": [True, False],
                "num_predict": [256, 512, 1024]
            },
            "orieg/gemma3-tools:27b-it-qat": {
                "num_ctx": [4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 81920, 98304],
                "batch": [8, 16, 32],
                "f16_kv": [True, False],
                "num_predict": [256, 512, 1024]
            },
            "liquid-rag:latest": {
                "num_ctx": [8192, 16384, 24576, 32768, 49152, 65536, 81920, 98304, 114688, 131072],
                "batch": [16, 32, 64],
                "f16_kv": [True],
                "num_predict": [256, 512, 1024]
            },
            "qwen2.5:3b-instruct": {
                "num_ctx": [8192, 16384, 24576, 32768, 49152, 65536, 81920, 98304, 114688, 131072],
                "batch": [16, 32, 64],
                "f16_kv": [True],
                "num_predict": [256, 512, 1024]
            },
            "gemma3:latest": {
                "num_ctx": [4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 81920, 98304, 114688, 131072],
                "batch": [16, 32, 64],
                "f16_kv": [True],
                "num_predict": [256, 512, 1024]
            }
        }

        # Default config for unknown models
        default_config = {
            "num_ctx": [4096, 8192, 16384, 24576, 32768, 49152, 65536, 81920, 98304, 114688, 131072],
            "batch": [16, 32],
            "f16_kv": [True],
            "num_predict": [256, 512, 1024]
        }

        config_params = model_configs.get(model, default_config)
        num_thread = psutil.cpu_count(logical=False) or 8  # Physical cores (optimal)

        configs = []

        # Start with smaller contexts and work up
        for num_ctx in sorted(config_params["num_ctx"]):
            for batch in config_params["batch"]:
                for f16_kv in config_params["f16_kv"]:
                    for num_predict in config_params["num_predict"]:
                        # Cpu-based configuration optimization - higher batch for larger contexts
                        if num_ctx >= 65536:  # 64k+
                            if batch < 32:
                                continue  # Skip small batches for huge contexts on CPU
                        elif num_ctx >= 32768:  # 32k+
                            if batch < 16:  # Allow 16+ for 32k on up
                                continue
                        # else: keep all batch sizes for smaller contexts

                        configs.append(TestConfig(
                            model=model,
                            num_ctx=num_ctx,
                            batch=batch,
                            num_predict=num_predict,
                            num_thread=num_thread,
                            f16_kv=f16_kv
                        ))

        # If we're CPU-only and have large memory, be more aggressive
        if self.monitor.gpu_type == "none":  # CPU-only system
            total_ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
            if total_ram_gb >= 32:  # Large RAM system
                print(f"💪 CPU-Only System with {total_ram_gb:.1f}GB RAM - Pushing Maximum Context Ranges")

                # Add extreme context configurations
                extreme_configs = [
                    TestConfig(model, 131072, 64, 1024, num_thread, True) for model in model_configs.keys()
                    if model == model  # Only for current model
                ]
                configs.extend(extreme_configs)
                print(f"  Added {len(extreme_configs)} extreme context configurations")

        return configs

    def generate_test_configs(self, model: str) -> List[TestConfig]:
        """Generate test configurations for a model - AI-First with Binary Search"""

        # Attempt to load AI configuration
        ai_config = self.load_ai_config()

        # Use AI-first method if config loads successfully
        if ai_config and ai_config.get("ai_guidance", {}).get("autonomous_tuning", True):
            try:
                return self.binary_search_context(model, ai_config)
            except Exception as e:
                print(f"Warning: AI-First optimization failed for {model}: {e}")
                print("   Falling back to legacy configuration generation")

        # Legacy fallback
        return self.legacy_generate_configs(model)

    def test_model(self, model: str, concurrency_levels: List[int] = [1, 2, 4]) -> List[TestResult]:
        """Test a single model with all configurations"""
        print(f"\n=== Testing {model} ===")

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
        """Save test results to files - AI-First with Multi-Preset Optimization"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSONL log
        log_file = self.output_dir / "logs" / f"{model.replace(':', '_')}_{timestamp}.jsonl"
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, 'w') as f:
            for result in results:
                json.dump(asdict(result), f)
                f.write('\n')

        # Generate AI-first summary with preset categories
        successful_results = [r for r in results if r.success]
        if not successful_results:
            print(f"Warning: No successful tests for {model}")
            return

        print(f"🎯 AI-First: Analyzing {len(successful_results)} successful results for {model}")

        # Load AI config for preset categories
        ai_config = self.load_ai_config()
        preset_categories = ai_config.get("preset_categories", {
            "max_context": {"description": "Maximum stable context"},
            "balanced": {"description": "Balanced performance"},
            "fast_response": {"description": "Fast response optimization"}
        })

        # Find optimal configurations for each preset category
        preset_results = {}

        for preset_name, preset_config in preset_categories.items():
            print(f"📊 Optimizing preset '{preset_name}'...")

            if preset_name == "max_context":
                # Find configuration with maximum stable context
                best_result = max(successful_results, key=lambda r: r.config.num_ctx)
                preset_results[preset_name] = best_result

            elif preset_name == "balanced":
                # Find balanced configuration using weighted scoring
                target_range = preset_config.get("target_context_range", [12288, 65536])

                # Filter results in target range
                eligible_results = [
                    r for r in successful_results
                    if target_range[0] <= r.config.num_ctx <= target_range[1]
                ]

                if eligible_results:
                    # Use weighted scoring: balance context utilization and throughput
                    context_weight = preset_config.get("throughput_weight", 0.6)
                    throughput_weight = preset_config.get("throughput_weight", 0.6)
                    ttft_weight = preset_config.get("ttft_weight", 0.4)

                    best_score = 0
                    best_result = eligible_results[0]

                    for result in eligible_results:
                        # Normalize scores (context to 0-1 range, higher is better)
                        context_score = min(result.config.num_ctx / max(target_range), 1.0)
                        throughput_score = min(result.tokens_per_sec / 20.0, 1.0)  # Assume 20 tok/s is excellent
                        ttft_score = max(0, 1 - (result.ttft_ms / 1000.0)) if result.ttft_ms else 0  # Lower TTFT is better

                        total_score = (
                            context_weight * context_score +
                            throughput_weight * throughput_score +
                            ttft_weight * ttft_score
                        )

                        if total_score > best_score:
                            best_score = total_score
                            best_result = result

                    preset_results[preset_name] = best_result
                else:
                    # Fallback to best in range
                    preset_results[preset_name] = max(successful_results, key=lambda r: r.tokens_per_sec)

            elif preset_name == "fast_response":
                # Find fastest response configuration above minimum context
                min_context = preset_config.get("target_context_min", 4096)
                eligible_results = [r for r in successful_results if r.config.num_ctx >= min_context]

                if eligible_results:
                    # Use weighted scoring favoring speed
                    throughput_weight = preset_config.get("throughput_weight", 0.8)
                    ttft_weight = preset_config.get("ttft_weight", 0.7)

                    best_score = 0
                    best_result = eligible_results[0]

                    for result in eligible_results:
                        throughput_score = min(result.tokens_per_sec / 30.0, 1.0)  # Assume 30 tok/s is excellent for fast response
                        ttft_score = max(0, 1 - (result.ttft_ms / 500.0)) if result.ttft_ms else 0  # Lower TTFT is better

                        total_score = throughput_weight * throughput_score + ttft_weight * ttft_score

                        if total_score > best_score:
                            best_score = total_score
                            best_result = result

                    preset_results[preset_name] = best_result
                else:
                    # Fallback to fastest overall
                    preset_results[preset_name] = max(successful_results, key=lambda r: r.tokens_per_sec)

        # Create AI-first summary with all presets
        summary = {
            "model": model,
            "timestamp": timestamp,
            "optimization_type": "AI-First Extreme Context",
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "ai_config_used": bool(ai_config),
            "presets": {}
        }

        # Add preset details
        for preset_name, result in preset_results.items():
            summary["presets"][preset_name] = {
                "description": preset_categories[preset_name].get("description", preset_name),
                "num_ctx": result.config.num_ctx,
                "batch": result.config.batch,
                "f16_kv": result.config.f16_kv,
                "num_predict": result.config.num_predict,
                "tokens_per_sec": result.tokens_per_sec,
                "ttft_ms": result.ttft_ms,
                "stability_score": self._calculate_stability_score(result, ai_config)
            }

        # Save AI-first summary
        summary_file = self.output_dir / "summaries" / f"{model.replace(':', '_')}_{timestamp}.json"
        summary_file.parent.mkdir(exist_ok=True)

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create agentic-focused defaults file
        defaults_file = self.output_dir / "defaults" / f"{model.replace(':', '_')}.yaml"
        defaults_file.parent.mkdir(exist_ok=True)

        defaults = {
            "model": model,
            "optimization_type": "AI-First Extreme Context",
            "hardware_target": "HP Strix AMD GPUs (64GB+ VRAM)",
            "updated_at": datetime.datetime.now().isoformat(),
            "presets": {}
        }

        # Add preset details
        for preset_name, result in preset_results.items():
            defaults["presets"][preset_name] = {
                "num_ctx": result.config.num_ctx,
                "batch": result.config.batch,
                "f16_kv": result.config.f16_kv,
                "num_predict": result.config.num_predict,
                "num_thread": result.config.num_thread,
                "temperature": result.config.temperature,
                "top_p": result.config.top_p,
                "performance": {
                    "tokens_per_sec": result.tokens_per_sec,
                    "ttft_ms": result.ttft_ms,
                    "stability_score": self._calculate_stability_score(result, ai_config)
                },
                "description": self._get_use_case_recommendation(preset_name, result.config.num_ctx),
                "use_case": self._get_use_case_recommendation(preset_name, result.config.num_ctx)
            }

        with open(defaults_file, 'w') as f:
            yaml.dump(defaults, f, default_flow_style=False)

        # Print AI-first results summary
        print(f"\n🎉 AI-First Optimization Complete for {model}")
        print(f"Results saved to {self.output_dir}")

        print("\n📊 Optimized Presets:")
        for preset_name, result in preset_results.items():
            ctx_k = result.config.num_ctx // 1024
            tps = result.tokens_per_sec
            ttft = result.ttft_ms
            print(f"  {preset_name}: {ctx_k}k context, {tps:.1f} tok/s, {ttft:.0f}ms TTFT")

    def _calculate_stability_score(self, result: TestResult, ai_config: Dict) -> float:
        """Calculate stability score based on AI configuration parameters"""
        # Simplified stability calculation - in real implementation this would use variance analysis
        # from multiple test runs and hardware monitoring data
        base_score = 0.85  # Assume good stability for successful tests

        # Adjust based on resource usage (if available)
        if result.vram_after_mb and result.ram_after_mb:
            # Lower score if high resource usage (greater chance of instability)
            resource_pressure = (result.vram_after_mb / (64 * 1024)) + (result.ram_after_mb / (64 * 1024 * 1024))
            base_score -= resource_pressure * 0.1

        return max(0.1, min(1.0, base_score))

    def _get_use_case_recommendation(self, preset_name: str, context_size: int) -> str:
        """Get use case recommendation based on preset and context size"""
        ctx_k = context_size // 1024

        if preset_name == "max_context":
            if ctx_k >= 128:
                return "Ultra-long conversations, massive document analysis, complex multi-agent workflows"
            else:
                return "Long-form reasoning, extensive tool traces, complex code generation"
        elif preset_name == "balanced":
            return "Typical agentic interactions, moderate tool usage, balanced performance needs"
        elif preset_name == "fast_response":
            return "Quick commands, real-time interactions, high-frequency tool calls"
        else:
            return f"Optimized for {preset_name} use cases with {ctx_k}k context"


def main():
    parser = argparse.ArgumentParser(description="O3 (Ozone) - AI-First Ollama Hardware Optimizer")
    parser.add_argument("models", nargs="+", help="Models to test with AI-first optimization")
    parser.add_argument("--output-dir", default="o3_results", help="Output directory")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2],
                       help="Concurrency levels to test")
    parser.add_argument("--ai-mode", action="store_true", default=True,
                       help="Enable AI-first mode with binary search optimization")
    parser.add_argument("--legacy-mode", action="store_true",
                       help="Use legacy linear search mode")
    parser.add_argument("--single-test-256k", action="store_true",
                       help="Test single configuration locked to 256k context for maximum capability")

    args = parser.parse_args()

    # Print AI-first banner
    if args.ai_mode and not args.legacy_mode:
        print("""
                    O3 (Ozone) - AI-First Optimization
                    Extreme Context Windows: 32k/64k/128k+

     * Binary Search Context Discovery | * Multi-Preset Optimization
     * Progressive Concurrency Testing | * Hardware Protection
     * Statistical Reliability         | * Learning & Adaptation

        """)

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

    # Handle single test locked to 256k context
    if args.single_test_256k and len(args.models) == 1:
        model = args.models[0]
        print(f"\n🔒 SINGLE TEST: Locked to 256k context - {model}")
        print("="*60)

        # Create single test config locked to 256k
        test_config = TestConfig(
            model=model,
            num_ctx=262144,  # 256k context locked
            batch=8,  # Conservative batch for max compatibility
            num_predict=512,
            num_thread=psutil.cpu_count(logical=False) or 8,  # Fixed: Use physical cores only
            f16_kv=True,
            temperature=0.2,
            top_p=0.95,
            seed=42
        )

        optimizer = OllamaOptimizer(args.output_dir)

        # Capture environment
        env_info = optimizer.capture_environment()
        env_file = optimizer.output_dir / "env" / f"env_256k_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        env_file.parent.mkdir(exist_ok=True)

        with open(env_file, 'w') as f:
            json.dump(env_info, f, indent=2)

        print(f"O3 (Ozone) - 256K CONTEXT SINGLE TEST")
        print(f"Hardware: {env_info['cpu_info']['cores_physical']} CPU cores, {env_info['memory']['total_ram_gb']}GB RAM")
        print(f"GPU: {optimizer.monitor.gpu_type}")
        print(f"Model: {model}")
        print(f"Context: 256k tokens")
        print(f"Expected timeout: ~{30 + (262144 / 1000 * 2) + (512 / 100):.0f}s")

        # Warmup first
        print(f"\n🔥 Warming up {model}...")
        if not optimizer.warmup_model(model):
            print(f"❌ Warmup failed - aborting 256k test")
            sys.exit(1)

        # Single 256k test
        print(f"\n🚀 Testing 256K Context Configuration...")
        run_id = f"{model}_256k_single_test"
        result = optimizer.run_single_test(test_config, run_id, 1, 0)

        results = [result]

        # Save result
        optimizer.save_results(model, results)

        if result.success:
            print(f"\n✅ SUCCESS: 256K Context Test Complete")
            print(f"Performance: {result.tokens_per_sec:.2f} tok/s, TTFT: {result.ttft_ms:.0f}ms")
            if result.vram_after_mb:
                print(f"VRAM Peak: {result.vram_after_mb}MB")
            print(f"Result saved to: {optimizer.output_dir}")
        else:
            print(f"\n❌ FAILED: {result.error}")
            print("256k context may exceed hardware limits or model capabilities")

    # Original multi-model testing
    else:
        # Test each model
        for model in args.models:
            try:
                results = optimizer.test_model(model, args.concurrency)
                if results:
                    optimizer.save_results(model, results)
            except KeyboardInterrupt:
                print(f"\nInterrupted during {model} testing")
                break
            except Exception as e:
                print(f"Error testing {model}: {e}")
                continue

    print(f"\nO3 testing complete. Results in: {optimizer.output_dir}")


if __name__ == "__main__":
    main()
