"""Swarm Concurrency Tester for Liquid Models

Optimizes small models for maximum concurrent instances in swarm deployments.
Focuses on minimal context + maximum concurrency for agentic swarms.
"""

import json
import time
import datetime
import concurrent.futures
import threading
from pathlib import Path
from typing import List, Dict, Optional

from ..core.optimizer import OllamaOptimizer, TestResult
from ..core.hardware_monitor import HardwareMonitor


class SwarmTester:
    """Specialized tester for liquid model swarm concurrency optimization"""

    def __init__(self, output_dir: str = "results/current"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.monitor = HardwareMonitor()

        # Swarm-specific settings
        self.swarm_prompt = "Remember this simple fact: AI agents work together to solve problems."
        self.concurrency_levels = [8, 16, 24, 32, 40, 48, 50, 56, 64]  # Push hard for swarm limits

        # Ultra-minimal config for speed
        self.swarm_config = {
            "num_ctx": 4096,      # Minimal context for speed
            "batch": 1,           # Smallest batch for minimal memory
            "num_predict": 32,    # Very short responses
            "num_thread": 1,      # Single thread per agent
            "f16_kv": True,
            "temperature": 0.1,  # Very low temperature for consistency
            "top_p": 0.9
        }

        print("🐝 Swarm Concurrency Tester initialized for liquid models")

    def run_swarm_concurrency_test(self, model: str = "liquid-rag:latest") -> Dict:
        """Push the limits of concurrent liquid model instances"""

        print(f"\n🕵️ SWARM CONCURRENCY TEST: {model}")
        print("=" * 80)
        print("Goal: Maximize concurrent 'quick memory agents' for swarm behavior")
        print("Approach: Minimal context + extreme concurrency levels")
        print(f"Configuration: {self.swarm_config}")
        print()

        # Warmup single instance
        print("🔥 Warming up single instance...")
        if not self.warmup_swarm_agent(model):
            print("❌ Swarm warmup failed")
            return {}

        swarm_results = {
            "model": model,
            "test_type": "swarm_concurrency_optimization",
            "timestamp": datetime.datetime.now().isoformat(),
            "config": self.swarm_config,
            "concurrency_tests": {}
        }

        max_stable_concurrency = 0
        peak_throughput = 0
        optimal_concurrency = 0

        for concurrency in self.concurrency_levels:
            print(f"\n{'='*20} Testing {concurrency} concurrent agents {'='*20}")

            try:
                # Test current concurrency level
                level_results = self.run_concurrency_level(model, concurrency)

                if level_results["success_rate"] >= 0.8:  # 80% success threshold
                    max_stable_concurrency = concurrency
                    total_throughput = level_results["total_throughput"]
                    peak_throughput = max(peak_throughput, total_throughput)

                    # Track optimal concurrency (peak efficiency)
                    if total_throughput > peak_throughput * 0.9:  # Within 90% of peak
                        optimal_concurrency = concurrency

                    print(f"✅ STABLE: {concurrency} agents, {total_throughput:.1f} total tok/s")

                elif level_results["success_rate"] >= 0.5:  # Partial success
                    print(f"⚠️  PARTIAL: {concurrency} agents, {level_results['success_rate']:.1%} success")
                    if max_stable_concurrency == 0:  # Allow partial if we haven't found stable yet
                        max_stable_concurrency = concurrency

                else:
                    print(f"❌ UNSTABLE: {concurrency} agents, {level_results['success_rate']:.1%} success")
                    break  # Stop at first unstable level

                swarm_results["concurrency_tests"][str(concurrency)] = level_results

            except Exception as e:
                print(f"💥 ERROR testing {concurrency} agents: {e}")
                swarm_results["concurrency_tests"][str(concurrency)] = {
                    "error": str(e),
                    "success_rate": 0.0,
                    "total_throughput": 0.0
                }
                break

        # Calculate final swarm metrics
        swarm_results.update({
            "max_stable_concurrency": max_stable_concurrency,
            "optimal_concurrency": optimal_concurrency,
            "peak_total_throughput": peak_throughput,
            "throughput_per_agent": peak_throughput / max_stable_concurrency if max_stable_concurrency > 0 else 0,
            "resource_efficiency": peak_throughput / max_stable_concurrency if max_stable_concurrency > 0 else 0
        })

        # Save results
        self.save_swarm_results(swarm_results, model)

        print(f"\n🎉 SWARM OPTIMIZATION COMPLETE")
        print(f"Maximum Stable Concurrency: {max_stable_concurrency} agents")
        print(f"Peak Total Throughput: {peak_throughput:.1f} tok/s")
        print(f"Optimal Swarm Size: {optimal_concurrency} agents")
        print(f"Avg Throughput per Agent: {peak_throughput / max_stable_concurrency:.1f} tok/s" if max_stable_concurrency > 0 else "")

        return swarm_results

    def run_concurrency_level(self, model: str, concurrency: int) -> Dict:
        """Run test at specific concurrency level"""
        print(f"Testing {concurrency} concurrent liquid agents...")

        start_time = time.time()
        results = []

        def run_single_agent(agent_id: int):
            """Run single agent instance"""
            try:
                result = self.run_swarm_agent(model, agent_id, concurrency)
                return result
            except Exception as e:
                return {
                    "agent_id": agent_id,
                    "success": False,
                    "error": str(e),
                    "tokens_per_sec": 0,
                    "ttft_ms": 0
                }

        # Launch concurrent agents
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_single_agent, i) for i in range(concurrency)]
            agent_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time
        successful_agents = [r for r in agent_results if r.get("success", False)]

        success_rate = len(successful_agents) / concurrency
        total_throughput = sum(r.get("tokens_per_sec", 0) for r in successful_agents)
        avg_ttft = sum(r.get("ttft_ms", 0) for r in successful_agents) / len(successful_agents) if successful_agents else 0

        return {
            "concurrency_level": concurrency,
            "total_agents": concurrency,
            "successful_agents": len(successful_agents),
            "success_rate": success_rate,
            "total_throughput": total_throughput,
            "avg_ttft_ms": avg_ttft,
            "total_test_time": total_time,
            "agent_results": agent_results
        }

    def run_swarm_agent(self, model: str, agent_id: int, total_concurrency: int) -> Dict:
        """Run single swarm agent instance"""
        import requests

        agent_prompt = f"Agent {agent_id}/{total_concurrency}: {self.swarm_prompt}"

        api_data = {
            "model": model,
            "prompt": agent_prompt,
            "stream": False,
            "options": self.swarm_config
        }

        start_time = time.time()

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=api_data,
                timeout=30  # Swarm agents need to be fast
            )

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            result = response.json()
            response_text = result.get('response', '')

            if len(response_text) < 3:  # Too short response
                raise Exception(f"Response too short: {len(response_text)} chars")

            # Calculate metrics
            total_time = time.time() - start_time
            output_tokens = len(response_text.split())
            tokens_per_sec = output_tokens / total_time if total_time > 0 else 0

            return {
                "agent_id": agent_id,
                "success": True,
                "tokens_per_sec": tokens_per_sec,
                "ttft_ms": total_time * 1000,
                "output_tokens": output_tokens,
                "response_text": response_text[:100]  # Sample for debugging
            }

        except Exception as e:
            return {
                "agent_id": agent_id,
                "success": False,
                "error": str(e),
                "tokens_per_sec": 0,
                "ttft_ms": 0
            }

    def warmup_swarm_agent(self, model: str) -> bool:
        """Quick warmup for swarm testing"""
        import requests

        try:
            api_data = {
                "model": model,
                "prompt": "Ready",
                "stream": False,
                "options": self.swarm_config
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=api_data,
                timeout=10
            )

            return response.status_code == 200

        except requests.exceptions.RequestException:
            return False

    def save_swarm_results(self, results: Dict, model: str):
        """Save swarm optimization results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        filename = f"swarm_optimization_{model.replace(':', '_')}_{timestamp}"
        results_file = self.output_dir / "summaries" / f"{filename}.json"
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save optimized swarm config
        config_file = self.output_dir / "defaults" / f"{filename}_config.yaml"
        config_file.parent.mkdir(exist_ok=True)

        import yaml
        swarm_config = {
            "model": model,
            "optimization_target": "swarm_concurrency_maximization",
            "optimal_concurrency": results.get("optimal_concurrency", 8),
            "max_stable_concurrency": results.get("max_stable_concurrency", 8),
            "configuration": self.swarm_config,
            "performance": {
                "peak_throughput_total": results.get("peak_total_throughput", 0),
                "throughput_per_agent": results.get("throughput_per_agent", 0),
                "resource_efficiency": results.get("resource_efficiency", 0)
            },
            "generated_at": timestamp,
            "use_case": "Concurrent quick memory agents for swarm intelligence"
        }

        with open(config_file, 'w') as f:
            yaml.dump(swarm_config, f, default_flow_style=False)

        print(f"📁 Swarm results saved to {self.output_dir}")


# CLI integration
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test swarm concurrency for liquid models")
    parser.add_argument("--model", default="liquid-rag:latest", help="Model to test")
    parser.add_argument("--output-dir", default="results/current", help="Output directory")

    args = parser.parse_args()

    swarm_tester = SwarmTester(args.output_dir)
    results = swarm_tester.run_swarm_concurrency_test(args.model)

    print("\nSwarm test complete!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
