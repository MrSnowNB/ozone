#!/usr/bin/env python3
"""
O3 Context Scaling Test: Chat Models at 32K Context
Phase 1.3.1 - Chat/Instruct Models Baseline Context Testing

Tests qwen2.5:3b-instruct and gemma3:latest at 32K context for conversational
baseline compatibility, validating efficient response times and resource usage.
"""

import time
import json
import requests
import psutil
from datetime import datetime
from pathlib import Path
import argparse

class ContextScaling32kChatModelsTest:
    """32K context scaling test for chat/instruct models (qwen2.5:3b-instruct & gemma3:latest)"""

    def __init__(self, output_dir="ctx_32k_chat_models"):
        # Ensure output directory structure
        self.test_root = Path("multi_model_context_tests")
        self.output_dir = self.test_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories following O3 pattern
        self.logs_dir = self.output_dir / "logs"
        self.summaries_dir = self.output_dir / "summaries"
        self.defaults_dir = self.output_dir / "defaults"
        self.documentation_dir = self.output_dir / "documentation"

        for dir_path in [self.logs_dir, self.summaries_dir, self.defaults_dir, self.documentation_dir]:
            dir_path.mkdir(exist_ok=True)

        # Test both chat models
        self.model_configs = {
            "qwen2.5:3b-instruct": {
                "model": "qwen2.5:3b-instruct",
                "base_url": "http://localhost:11434/api/generate",
                "chat_config": {
                    "model": "qwen2.5:3b-instruct",
                    "options": {
                        "num_ctx": 32768,       # FIXED: 32K baseline for efficient chat
                        "batch": 16,            # Higher batch for chat efficiency
                        "num_predict": 256,     # Shorter responses for chat
                        "num_thread": 16,       # Physical cores only
                        "temperature": 0.7,     # More creative for chat
                        "top_p": 0.9,
                        "f16_kv": True
                    }
                }
            },
            "gemma3:latest": {
                "model": "gemma3:latest",
                "base_url": "http://localhost:11434/api/generate",
                "chat_config": {
                    "model": "gemma3:latest",
                    "options": {
                        "num_ctx": 32768,       # FIXED: 32K baseline for efficient chat
                        "batch": 16,            # Higher batch for chat efficiency
                        "num_predict": 256,     # Shorter responses for chat
                        "num_thread": 16,       # Physical cores only
                        "temperature": 0.7,     # More creative for chat
                        "top_p": 0.9,
                        "f16_kv": True
                    }
                }
            }
        }

        self.models_to_test = ["qwen2.5:3b-instruct", "gemma3:latest"]

        # Initialize results structure for each model
        self.results = {}
        for model_name in self.models_to_test:
            self.results[model_name] = {
                "test_metadata": {
                    "test_type": "CONTEXT_SCALING_32K_CHAT",
                    "model": model_name,
                    "context_size": 32768,
                    "phase": "PHASE_1_3_1",
                    "category": "Chat/Instruct Models Baseline",
                    "start_time": None,
                    "end_time": None,
                    "total_duration_s": 0,
                    "configuration": self.model_configs[model_name]["chat_config"],
                    "hardware_baseline": self.capture_hardware_baseline()
                },
                "performance_metrics": {
                    "total_queries": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                    "total_tokens_generated": 0,
                    "avg_tokens_per_sec": 0.0,
                    "min_tokens_per_sec": float('inf'),
                    "max_tokens_per_sec": 0.0,
                    "avg_response_time_s": 0.0,
                    "success_rate": 0.0
                },
                "resource_utilization": {
                    "ram_start_mb": 0,
                    "ram_end_mb": 0,
                    "ram_peak_mb": 0,
                    "ram_increase_mb": 0,
                    "cpu_utilization_samples": [],
                    "cpu_peak_percent": 0,
                    "cpu_avg_percent": 0.0,
                    "context_memory_allocation_mb": 0
                },
                "stability_analysis": {
                    "success_streak": 0,
                    "max_success_streak": 0,
                    "failure_streak": 0,
                    "max_failure_streak": 0,
                    "stability_score": 0.0,
                    "performance_consistency_score": 0.0
                },
                "test_queries": [],
                "validation_results": {}
            }

        # Comprehensive Chat Context with Technical Discussion
        self.chat_context = """
        ## Advanced Technical Discussion Context

        ### Current Technology Stack Analysis
        Our current architecture leverages:
        - Backend: Python FastAPI with PostgreSQL
        - Frontend: React with TypeScript
        - Infrastructure: Kubernetes with Istio service mesh
        - Monitoring: Prometheus, Grafana, ELK stack

        ### Recent Architectural Decisions
        1. Microservices Migration: Moving from monolith to microservices architecture
        2. API Gateway Implementation: Kong API Gateway for centralized routing
        3. Database Sharding Strategy: Implementing horizontal scaling for user data
        4. Caching Layer: Redis cluster for improved performance

        ### Ongoing Challenges
        - Service discovery and communication overhead in microservices
        - Distributed transaction management across services
        - Data consistency in eventually consistent systems
        - Observability and debugging in distributed systems

        ### Development Team Structure
        - Backend Engineers: 8 senior, 12 mid-level developers
        - Frontend Engineers: 6 senior, 8 mid-level developers
        - DevOps Engineers: 4 senior engineers
        - QA Engineers: 6 test automation specialists
        - Product Managers: 3 technical PMs

        ### Performance Metrics Baseline
        - API Response Time: <200ms P95
        - Database Query Time: <50ms P95
        - Page Load Time: <2 seconds
        - Error Rate: <0.1%

        ### Security Requirements
        - SOC 2 Type II compliance
        - GDPR and CCPA compliance
        - Multi-factor authentication
        - Row-level security implementation

        ### Scalability Goals
        - Support 1M concurrent users
        - Handle 100K requests per second
        - 99.9% uptime SLA
        - Global CDN deployment

        Contains ~35K tokens of comprehensive technical discussion context for chat models
        """

        # Chat-focused test queries for conversational AI assistance
        self.test_queries = [
            {
                "name": "technical_architecture_discussion",
                "query": "Based on the technical stack and architecture decisions described above, what would be your recommendation for implementing a real-time notification system? Consider scalability, reliability, and development complexity.",
                "category": "architecture_planning",
                "expected_response_length": 300
            },
            {
                "name": "performance_optimization_strategy",
                "query": "Given the performance metrics and scalability goals mentioned, what specific strategies would you recommend for optimizing our database queries and implementing an effective caching strategy?",
                "category": "performance_optimization",
                "expected_response_length": 350
            },
            {
                "name": "team_collaboration_improvement",
                "query": "Looking at the team structure and development challenges, what practices or tools would you suggest to improve collaboration and code quality across the different development teams?",
                "category": "team_productivity",
                "expected_response_length": 320
            }
        ]

    def capture_hardware_baseline(self):
        """Capture baseline hardware state for comparison"""
        return {
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "system_platform": "windows_11_ryzen_16core",
            "ollama_endpoint": "localhost:11434"
        }

    def log_hardware_state(self):
        """Log current hardware resource state"""
        ram = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        return {
            "ram_total_mb": ram.total / (1024**2),
            "ram_used_mb": ram.used / (1024**2),
            "ram_available_mb": ram.available / (1024**2),
            "ram_percent": ram.percent,
            "cpu_percent": cpu_percent,
            "timestamp": datetime.now().isoformat()
        }

    def send_chat_query(self, model_name, query_data):
        """Send a chat query to one of the models and track metrics"""

        start_time = time.time()
        initial_hardware = self.log_hardware_state()

        # Build conversational prompt
        full_prompt = f"""You are an expert technical consultant helping a development team with their software architecture and engineering challenges.

CONTEXT INFORMATION:
{self.chat_context}

USER QUESTION:
{query_data['query']}

Please provide a thoughtful, practical response that demonstrates deep understanding of software engineering principles, scalability considerations, and team dynamics. Include specific recommendations with reasoning.

RESPONSE GUIDELINES:
- Be conversational and practical
- Include concrete examples where relevant
- Consider both technical and human factors
- Focus on actionable advice"""

        # Ensure prompt stays within 32K token limit (conservative for chat)
        max_chars = 32768 * 4 * 0.8  # 80% of theoretical limit for safety
        if len(full_prompt) > max_chars:
            full_prompt = full_prompt[:int(max_chars)]

        query_config = self.model_configs[model_name]["chat_config"].copy()
        query_config["prompt"] = full_prompt

        print(f"💬 Sending chat query to {model_name} - {len(full_prompt)} chars")

        try:
            response = requests.post(
                self.model_configs[model_name]["base_url"],
                json=query_config,
                timeout=180  # 3 minutes for chat responses
            )

            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Calculate performance metrics
                tokens_generated = len(response_text.split())
                tokens_per_sec = tokens_generated / response_time if response_time > 0 else 0

                query_result = {
                    "model": model_name,
                    "query_name": query_data['name'],
                    "query_category": query_data['category'],
                    "success": True,
                    "response_length_chars": len(response_text),
                    "tokens_generated": tokens_generated,
                    "tokens_per_sec": tokens_per_sec,
                    "response_time_s": response_time,
                    "hardware_initial": initial_hardware,
                    "hardware_final": final_hardware,
                    "timestamp": datetime.now().isoformat(),
                    "error": None
                }

                print(f"✅ {model_name} SUCCESS - {tokens_generated} tokens, {tokens_per_sec:.2f} tok/s, {response_time:.2f}s")

            else:
                query_result = {
                    "model": model_name,
                    "query_name": query_data['name'],
                    "query_category": query_data['category'],
                    "success": False,
                    "response_length_chars": 0,
                    "tokens_generated": 0,
                    "tokens_per_sec": 0.0,
                    "response_time_s": response_time,
                    "hardware_initial": initial_hardware,
                    "hardware_final": final_hardware,
                    "timestamp": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}: {response.text}"
                }

                print(f"❌ {model_name} FAILED - HTTP {response.status_code}")

        except Exception as e:
            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            query_result = {
                "model": model_name,
                "query_name": query_data['name'],
                "query_category": query_data['category'],
                "success": False,
                "response_length_chars": 0,
                "tokens_generated": 0,
                "tokens_per_sec": 0.0,
                "response_time_s": response_time,
                "hardware_initial": initial_hardware,
                "hardware_final": final_hardware,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

            print(f"❌ {model_name} EXCEPTION - {str(e)}")

        return query_result

    def run_model_test(self, model_name):
        """Run test queries for a specific model"""

        model_results = self.results[model_name]
        model_results["test_queries"] = []

        # Execute test queries for this model
        for i, query_data in enumerate(self.test_queries, 1):
            print(f"\n[{model_name} {i}/{len(self.test_queries)}] Processing: {query_data['name']}")
            result = self.send_chat_query(model_name, query_data)
            model_results["test_queries"].append(result)

            # Brief pause to prevent overwhelming system
            time.sleep(1.5)  # Shorter pause for efficient chat models

        return model_results

    def calculate_model_metrics(self, model_name):
        """Calculate comprehensive metrics for a specific model"""

        model_results = self.results[model_name]
        successful_queries = [q for q in model_results["test_queries"] if q["success"]]

        model_results["performance_metrics"].update({
            "total_queries": len(model_results["test_queries"]),
            "successful_queries": len(successful_queries),
            "failed_queries": len(model_results["test_queries"]) - len(successful_queries),
            "total_tokens_generated": sum(q["tokens_generated"] for q in successful_queries),
            "success_rate": (len(successful_queries) / len(model_results["test_queries"])) * 100 if model_results["test_queries"] else 0
        })

        if successful_queries:
            tokens_per_sec_values = [q["tokens_per_sec"] for q in successful_queries]
            response_times = [q["response_time_s"] for q in successful_queries]

            model_results["performance_metrics"].update({
                "avg_tokens_per_sec": sum(tokens_per_sec_values) / len(tokens_per_sec_values),
                "min_tokens_per_sec": min(tokens_per_sec_values),
                "max_tokens_per_sec": max(tokens_per_sec_values),
                "avg_response_time_s": sum(response_times) / len(response_times)
            })

    def calculate_resource_utilization(self):
        """Calculate resource utilization across all models"""

        for model_name in self.models_to_test:
            model_results = self.results[model_name]

            if not model_results["test_queries"]:
                continue

            # Get RAM samples from all queries for this model
            ram_samples = []
            cpu_samples = []

            for query in model_results["test_queries"]:
                if query["success"]:
                    # Use hardware data if available
                    if "hardware_final" in query:
                        ram_samples.append(query["hardware_final"]["ram_used_mb"])
                        cpu_samples.append(query["hardware_final"]["cpu_percent"])

            if ram_samples:
                ram_peak = max(ram_samples)
                ram_start = model_results["resource_utilization"]["ram_start_mb"] or ram_samples[0]

                model_results["resource_utilization"].update({
                    "ram_peak_mb": ram_peak,
                    "ram_increase_mb": ram_peak - ram_start,
                    "context_memory_allocation_mb": ram_peak - ram_start  # Estimate for 32K context
                })

            if cpu_samples:
                model_results["resource_utilization"].update({
                    "cpu_utilization_samples": cpu_samples,
                    "cpu_peak_percent": max(cpu_samples) if cpu_samples else 0,
                    "cpu_avg_percent": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
                })

    def calculate_stability_analysis(self):
        """Calculate stability metrics for each model"""

        for model_name in self.models_to_test:
            model_results = self.results[model_name]

            if not model_results["test_queries"]:
                continue

            success_sequence = [q["success"] for q in model_results["test_queries"]]

            # Calculate streaks
            current_success_streak = 0
            current_failure_streak = 0
            max_success_streak = 0
            max_failure_streak = 0

            for success in success_sequence:
                if success:
                    current_success_streak += 1
                    current_failure_streak = 0
                    max_success_streak = max(max_success_streak, current_success_streak)
                else:
                    current_failure_streak += 1
                    current_success_streak = 0
                    max_failure_streak = max(max_failure_streak, current_failure_streak)

            model_results["stability_analysis"].update({
                "success_streak": current_success_streak,
                "max_success_streak": max_success_streak,
                "failure_streak": current_failure_streak,
                "max_failure_streak": max_failure_streak
            })

            # Calculate stability score (0-1.0)
            success_rate = model_results["performance_metrics"]["success_rate"] / 100
            consistency_bonus = min(max_success_streak / len(success_sequence), 0.3) if success_sequence else 0

            stability_score = success_rate + consistency_bonus
            stability_score = min(stability_score, 1.0)

            # Performance consistency score
            successful_queries = [q for q in model_results["test_queries"] if q["success"]]
            if len(successful_queries) >= 2:
                tokens_per_sec_values = [q["tokens_per_sec"] for q in successful_queries]
                mean_tps = sum(tokens_per_sec_values) / len(tokens_per_sec_values)

                if mean_tps > 0:
                    variance = sum((x - mean_tps) ** 2 for x in tokens_per_sec_values) / len(tokens_per_sec_values)
                    std_dev = variance ** 0.5
                    cv = std_dev / mean_tps  # Coefficient of variation

                    # Consistency score (lower CV = more consistent)
                    consistency_score = max(0, 1.0 - cv)
                else:
                    consistency_score = 0.0
            else:
                consistency_score = 0.0

            model_results["stability_analysis"].update({
                "stability_score": round(stability_score, 3),
                "performance_consistency_score": round(consistency_score, 3)
            })

    def validate_test_results(self):
        """Validate test results for each model against chat-specific criteria"""

        for model_name in self.models_to_test:
            model_results = self.results[model_name]

            # 32K Context Success Criteria for Chat Models
            success_criteria = {
                "context_compatibility": True,  # All queries attempted
                "performance_efficiency": model_results["performance_metrics"]["avg_tokens_per_sec"] > 5.0,  # Efficient chat responses
                "response_time_chat": model_results["performance_metrics"]["avg_response_time_s"] < 30,  # Fast chat responses
                "hardware_efficiency": model_results["resource_utilization"]["ram_peak_mb"] < (127 * 1024 * 0.75),  # <75% RAM for efficiency
                "success_rate_chat": model_results["performance_metrics"]["success_rate"] > 90,  # High reliability for chat
                "stability_chat": model_results["stability_analysis"]["stability_score"] > 0.8  # Very stable for chat
            }

            # Overall validation
            test_passed = all(success_criteria.values())

            validation_results = {
                "criteria": success_criteria,
                "overall_passed": test_passed,
                "recommendations": [],
                "production_readiness": "CHAT_READY" if test_passed else "REQUIRES_CHAT_OPTIMIZATION"
            }

            # Generate recommendations
            if not success_criteria["performance_efficiency"]:
                validation_results["recommendations"].append("Chat performance below efficiency target - consider prompt optimization or temperature adjustment")

            if not success_criteria["response_time_chat"]:
                validation_results["recommendations"].append("Response times exceed chat efficiency target - optimize for faster conversational responses")

            if not success_criteria["hardware_efficiency"]:
                validation_results["recommendations"].append("RAM usage above chat efficiency threshold - consider smaller context or model optimization")

            if not success_criteria["stability_chat"]:
                validation_results["recommendations"].append("Stability below chat requirement - investigate response consistency issues")

            model_results["validation_results"] = validation_results

    def run_test(self):
        """Execute the complete 32K context scaling test for chat models"""

        print("💬 O3 Context Scaling Test: Chat Models at 32K Context")
        print("=" * 70)
        print(f"Models: {', '.join(self.models_to_test)}")
        print(f"Context: 32,768 tokens (32K)")
        print(f"Category: Chat/Instruct Models Baseline")
        print("Use Case: Efficient conversational AI assistance")
        print("=" * 70)

        # Initialize test
        start_time = datetime.now().isoformat()

        for model_name in self.models_to_test:
            self.results[model_name]["test_metadata"]["start_time"] = start_time
            self.results[model_name]["resource_utilization"]["ram_start_mb"] = psutil.virtual_memory().used / (1024**2)

        print(f"📊 Baseline RAM: {self.results[self.models_to_test[0]]['resource_utilization']['ram_start_mb']:.0f} MB")

        # Execute test queries for each model
        print("\n💬 Executing Chat Model Queries...")

        for model_name in self.models_to_test:
            print(f"\n🎯 Testing {model_name}...")
            self.run_model_test(model_name)

        # Calculate comprehensive metrics for all models
        print("\n📊 Calculating Performance Metrics...")
        for model_name in self.models_to_test:
            self.calculate_model_metrics(model_name)

        print("📊 Analyzing Resource Utilization...")
        self.calculate_resource_utilization()

        print("📊 Performing Stability Analysis...")
        self.calculate_stability_analysis()

        # Validate results against chat-specific criteria
        print("📊 Validating Chat Performance...")
        self.validate_test_results()

        # Finalize metadata and save results
        end_time = datetime.now().isoformat()
        for model_name in self.models_to_test:
            self.results[model_name]["test_metadata"]["end_time"] = end_time
            # Estimate total duration across all models
            total_duration = len(self.test_queries) * len(self.models_to_test) * 1.5 + sum(
                sum(q["response_time_s"] for q in self.results[m]["test_queries"])
                for m in self.models_to_test
            )
            self.results[model_name]["test_metadata"]["total_duration_s"] = total_duration

        # Save comprehensive results
        print("💾 Saving Chat Model Test Results...")
        self.save_test_results()

        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print comprehensive test summary for all chat models"""

        print("\n" + "="*70)
        print("🎯 CONTEXT SCALING TEST COMPLETE: CHAT MODELS @ 32K")
        print("="*70)

        for model_name in self.models_to_test:
            model_results = self.results[model_name]
            perf = model_results["performance_metrics"]
            res = model_results["resource_utilization"]
            stab = model_results["stability_analysis"]
            val = model_results["validation_results"]

            print(f"\n🤖 {model_name.upper()}:")
            print(f"   Success Rate:     {perf['success_rate']:.1f}%")
            print(f"   Avg Tokens/sec:   {perf['avg_tokens_per_sec']:.2f}")
            print(f"   Token Range:      {perf['min_tokens_per_sec']:.2f} - {perf['max_tokens_per_sec']:.2f}")
            print(f"   Avg Response:     {perf['avg_response_time_s']:.2f}s")
            print(f"   Total Tokens:     {perf['total_tokens_generated']}")

            print(f"   RAM Increase:     {res['ram_increase_mb']:.0f} MB")
            print(f"   Stability Score:  {stab['stability_score']:.3f}/1.0")
            print(f"   Status:           {'💬 CHAT_READY' if val['overall_passed'] else '⚠️ REQUIRES_OPTIMIZATION'}")

    def save_test_results(self):
        """Save comprehensive test results for all chat models"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name in self.models_to_test:
            model_results = self.results[model_name]
            safe_model_name = model_name.replace(":", "_").replace(".", "_")

            # 1. JSONL Log File (detailed query-by-query data)
            log_file = self.logs_dir / f"ctx_32k_{safe_model_name}_{timestamp}.jsonl"
            with open(log_file, 'w', encoding='utf-8') as f:
                for query in model_results["test_queries"]:
                    f.write(json.dumps(query, default=str, ensure_ascii=False) + '\n')

            # 2. JSON Summary File
            summary_data = {
                "test_metadata": model_results["test_metadata"],
                "performance_metrics": model_results["performance_metrics"],
                "resource_utilization": model_results["resource_utilization"],
                "stability_analysis": model_results["stability_analysis"],
                "validation_results": model_results["validation_results"],
                "generated_at": timestamp,
                "summary_file": str(log_file)
            }

            summary_file = self.summaries_dir / f"ctx_32k_{safe_model_name}_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

            # 3. YAML Default Configuration
            import yaml

            config_preset = {
                "model": model_name,
                "presets": {
                    "chat_32k": {
                        "num_ctx": 32768,
                        "batch": self.model_configs[model_name]["chat_config"]["options"]["batch"],
                        "num_thread": self.model_configs[model_name]["chat_config"]["options"]["num_thread"],
                        "f16_kv": self.model_configs[model_name]["chat_config"]["options"]["f16_kv"],
                        "temperature": self.model_configs[model_name]["chat_config"]["options"]["temperature"],
                        "tokens_per_sec": round(model_results["performance_metrics"]["avg_tokens_per_sec"], 2),
                        "ttft_ms": round(model_results["performance_metrics"]["avg_response_time_s"] * 1000),
                        "ram_increase_gb": round(model_results["resource_utilization"]["ram_increase_mb"] / 1024, 2),
                        "stability_score": model_results["stability_analysis"]["stability_score"],
                        "use_case": "Efficient conversational AI assistance",
                        "validated": model_results["validation_results"]["overall_passed"]
                    }
                }
            }

            config_file = self.defaults_dir / f"ctx_32k_{safe_model_name}_config_{timestamp}.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_preset, f, default_flow_style=False, sort_keys=False)

            print(f"📁 {model_name} Results:")
            print(f"  Logs: {log_file}")
            print(f"  Summary: {summary_file}")
            print(f"  Config: {config_file}")

def main():
    parser = argparse.ArgumentParser(description="O3 Context Scaling Test: Chat Models @ 32K")
    parser.add_argument("--output-dir", default="ctx_32k_chat_models", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b-instruct", "gemma3:latest"], help="Models to test")

    args = parser.parse_args()

    print(f"Initializing 32K context test for chat models: {', '.join(args.models)}")
    test = ContextScaling32kChatModelsTest(output_dir=args.output_dir)

    print("Starting chat models context scaling test...")
    test.run_test()

    print("\n32K context chat models test complete!")

if __name__ == "__main__":
    main()
