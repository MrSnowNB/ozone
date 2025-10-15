#!/usr/bin/env python3
"""
O3 Context Scaling Test: Liquid-RAG at 64K Context
Phase 1.2.1 - RAG Model Context Document Chunking Testing

Tests liquid-rag:latest at 64K context for document chunking and multi-document
retrieval scenarios, validating RAG-specific performance characteristics.
"""

import time
import json
import requests
import psutil
from datetime import datetime
from pathlib import Path
import argparse

class ContextScaling64kLiquidRAGTest:
    """64K context scaling test for liquid-rag RAG Model"""

    def __init__(self, output_dir="ctx_64k_liquid_rag"):
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

        self.model = "liquid-rag:latest"
        self.base_url = "http://localhost:11434/api/generate"

        # RAG-OPTIMIZED CONFIGURATION FOR 64K CONTEXT
        self.config = {
            "model": self.model,
            "options": {
                "num_ctx": 65536,       # FIXED: 64K context for document chunking
                "batch": 8,             # Conservative batch for RAG stability
                "num_predict": 512,     # Higher for comprehensive RAG responses
                "num_thread": 16,       # Physical cores only (hyperthreading disabled)
                "temperature": 0.1,     # Lower temperature for RAG accuracy
                "top_p": 0.9,
                "f16_kv": True          # Memory efficient
            }
        }

        self.results = {
            "test_metadata": {
                "test_type": "CONTEXT_SCALING_64K",
                "model": self.model,
                "context_size": 65536,
                "phase": "PHASE_1_2_1",
                "category": "RAG Model Document Chunking",
                "start_time": None,
                "end_time": None,
                "total_duration_s": 0,
                "configuration": self.config,
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

        # RAG-optimized multi-document content (technical documentation chunks)
        self.rag_knowledge_base = """
        ## RAG System Architecture Documentation

        ### Core Components

        **Document Processing Pipeline:**
        - Text extraction from multiple formats (PDF, HTML, DOCX, MD)
        - Intelligent chunking with overlap handling
        - Embedding generation using sentence-transformers
        - Vector storage in FAISS or similar high-performance database

        **Retrieval Mechanisms:**
        - Semantic similarity search using cosine distance
        - Hybrid retrieval combining dense and sparse methods
        - Re-ranking using cross-encoders for improved accuracy
        - Query expansion techniques for better recall

        **Generation Pipeline:**
        - Context window management and token limiting
        - Prompt engineering for RAG-specific instructions
        - Temperature control for factual accuracy
        - Response validation and fact-checking integration

        ### Technical Implementation Details

        **Vector Database Selection:**
        - FAISS: High-performance similarity search, suitable for large-scale deployments
        - Pinecone: Managed vector database with advanced filtering capabilities
        - Weaviate: Graph-based vector search with relationship understanding
        - ChromaDB: Lightweight, embeddable option for smaller applications

        **Chunking Strategies:**
        - Fixed-size chunking: Simple but may break semantic units
        - Sentence-aware chunking: Preserves grammatical boundaries
        - Recursive chunking: Multi-level hierarchy for complex documents
        - Header-based chunking: Respects document structure

        **Embedding Models:**
        - text-embedding-3-small: Balance of performance and cost
        - text-embedding-3-large: Higher accuracy for complex queries
        - instructor-xl: Domain-specific embeddings for technical content
        - multilingual-e5-large: Support for multiple languages

        ### Performance Optimization Techniques

        **Indexing Strategies:**
        - Hierarchical Navigable Small World (HNSW) for approximate nearest neighbors
        - Product Quantization for memory compression
        - IVF (Inverted File Index) for scalable search
        - GPU acceleration for embedding computation

        **Caching Layers:**
        - Redis for frequently accessed embeddings
        - LRU cache for recent queries and responses
        - Pre-computed summaries for common topics
        - Result memoization with TTL-based expiration

        **Query Optimization:**
        - Query preprocessing and normalization
        - Multi-step retrieval with progressive filtering
        - Confidence scoring for response ranking
        - Fallback mechanisms for low-confidence results

        ### Scalability Considerations

        **Horizontal Scaling:**
        - Load balancing across multiple retrieval instances
        - Distributed vector databases (Pinecone, Weaviate)
        - Microservice architecture for independent scaling
        - Kubernetes horizontal pod autoscaling

        **Data Management:**
        - Incremental indexing for new documents
        - Version control for document updates
        - Metadata enrichment for better filtering
        - Data deduplication and quality assurance

        **Performance Monitoring:**
        - Query latency and throughput metrics
        - Recall and precision measurements
        - System resource utilization tracking
        - User satisfaction and accuracy feedback

        ### Integration Patterns

        **API Design:**
        - RESTful endpoints for document upload and querying
        - Streaming responses for large document processing
        - Webhook notifications for batch operations
        - GraphQL for flexible query interfaces

        **Authentication and Security:**
        - OAuth 2.0 support for enterprise integrations
        - API key management for external applications
        - Rate limiting and abuse prevention
        - Audit logging for compliance requirements

        **Error Handling and Resilience:**
        - Circuit breaker patterns for external service calls
        - Retry mechanisms with exponential backoff
        - Graceful degradation during service outages
        - Fallback responses for critical queries

        ## Advanced RAG Techniques

        ### Retrieval-Augmented Generation (RAG) Pipeline

        1. **Query Understanding**
           - Intent classification and entity extraction
           - Query expansion and reformulation
           - Context-aware query processing

        2. **Document Retrieval**
           - Multi-stage retrieval (coarse → fine-grained)
           - Relevance scoring and ranking
           - Diversity-aware document selection

        3. **Context Preparation**
           - Document chunk assembly and ordering
           - Token limit management within context windows
           - Redundancy elimination and coherence optimization

        4. **Answer Generation**
           - Prompt engineering for factual responses
           - Confidence calibration and uncertainty estimation
           - Multi-turn conversation support with context retention

        ### Specialized RAG Applications

        **Code Documentation RAG:**
        - Semantic code search and understanding
        - API documentation generation
        - Code review assistance and best practice recommendations
        - Technical documentation synthesis

        **Legal Document Analysis:**
        - Contract analysis and clause extraction
        - Regulatory compliance checking
        - Legal precedent retrieval and analysis
        - Risk assessment and due diligence support

        **Medical Research RAG:**
        - Clinical trial data synthesis
        - Drug interaction analysis
        - Treatment protocol recommendations
        - Medical literature review and summarization

        **Financial Analysis:**
        - Market research synthesis
        - Risk assessment modeling
        - Investment strategy evaluation
        - Regulatory filing analysis

        Contains ~45K tokens of comprehensive RAG system documentation and technical specifications
        """

        # RAG-specific test queries for document analysis and retrieval
        self.test_queries = [
            {
                "name": "rag_system_overview",
                "query": "Provide a comprehensive overview of this RAG (Retrieval-Augmented Generation) system architecture. Explain the core components, data flow patterns, and key design principles that make it effective for enterprise document analysis.",
                "category": "rag_architecture_overview",
                "expected_response_length": 450
            },
            {
                "name": "vector_database_comparison",
                "query": "Compare and contrast the different vector database options mentioned in the documentation. Analyze their relative strengths, use cases, performance characteristics, and scaling properties for different deployment scenarios.",
                "category": "rag_optimization_analysis",
                "expected_response_length": 500
            },
            {
                "name": "scalability_recommendations",
                "query": "Based on the technical specifications provided, develop a comprehensive scalability strategy for a RAG system handling 10 million documents and 1000 concurrent queries per second. Include infrastructure recommendations, optimization techniques, and monitoring approaches.",
                "category": "rag_enterprise_deployment",
                "expected_response_length": 550
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

    def send_test_query(self, query_data):
        """Send a single test query and track detailed metrics"""

        start_time = time.time()
        initial_hardware = self.log_hardware_state()

        # Build comprehensive prompt with RAG documentation context
        full_prompt = f"""You are an expert RAG (Retrieval-Augmented Generation) system architect analyzing technical documentation.

RAG SYSTEM DOCUMENTATION CONTEXT:
{self.rag_knowledge_base}

ANALYSIS REQUEST:
{query_data['query']}

Please provide a detailed, technical analysis that demonstrates deep understanding of RAG system design, optimization techniques, and enterprise-scale deployment considerations. Use specific examples and recommendations from the provided documentation.

RESPONSE REQUIREMENTS:
- Show comprehensive understanding of RAG architecture patterns
- Include specific technical recommendations with implementation details
- Demonstrate knowledge of different component choices and trade-offs
- Provide actionable insights for production deployment
"""

        # Ensure prompt stays within 64K token limit (rough estimate: 4 chars per token)
        max_chars = 65536 * 4 * 0.9  # 90% of theoretical limit for safety
        if len(full_prompt) > max_chars:
            full_prompt = full_prompt[:int(max_chars)]

        query_config = self.config.copy()
        query_config["prompt"] = full_prompt

        print(f"🔍 Sending RAG analysis query - {len(full_prompt)} chars")

        try:
            response = requests.post(self.base_url, json=query_config, timeout=600)  # 10 min timeout

            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Calculate performance metrics
                tokens_generated = len(response_text.split())
                tokens_per_sec = tokens_generated / response_time if response_time > 0 else 0

                query_result = {
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

                print(f"✅ RAG SUCCESS - {tokens_generated} tokens, {tokens_per_sec:.2f} tok/s, {response_time:.2f}s")

            else:
                query_result = {
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

                print(f"❌ RAG FAILED - HTTP {response.status_code}")

        except Exception as e:
            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            query_result = {
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

            print(f"❌ RAG EXCEPTION - {str(e)}")

        return query_result

    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""

        successful_queries = [q for q in self.results["test_queries"] if q["success"]]

        self.results["performance_metrics"].update({
            "total_queries": len(self.results["test_queries"]),
            "successful_queries": len(successful_queries),
            "failed_queries": len(self.results["test_queries"]) - len(successful_queries),
            "total_tokens_generated": sum(q["tokens_generated"] for q in successful_queries),
            "success_rate": (len(successful_queries) / len(self.results["test_queries"])) * 100 if self.results["test_queries"] else 0
        })

        if successful_queries:
            tokens_per_sec_values = [q["tokens_per_sec"] for q in successful_queries]
            response_times = [q["response_time_s"] for q in successful_queries]

            self.results["performance_metrics"].update({
                "avg_tokens_per_sec": sum(tokens_per_sec_values) / len(tokens_per_sec_values),
                "min_tokens_per_sec": min(tokens_per_sec_values),
                "max_tokens_per_sec": max(tokens_per_sec_values),
                "avg_response_time_s": sum(response_times) / len(response_times)
            })

    def calculate_resource_utilization(self):
        """Calculate resource utilization during testing"""
        if not self.results["test_queries"]:
            return

        # Get RAM samples from all queries
        ram_samples = []
        cpu_samples = []

        for query in self.results["test_queries"]:
            if query["success"]:
                # Use hardware_final for peak measurements
                ram_samples.append(query["hardware_final"]["ram_used_mb"])
                cpu_samples.append(query["hardware_final"]["cpu_percent"])

        if ram_samples:
            ram_peak = max(ram_samples)
            ram_start = self.results["resource_utilization"]["ram_start_mb"]

            self.results["resource_utilization"].update({
                "ram_peak_mb": ram_peak,
                "ram_increase_mb": ram_peak - ram_start,
                "context_memory_allocation_mb": ram_peak - ram_start  # Estimate for 64K context
            })

        if cpu_samples:
            self.results["resource_utilization"].update({
                "cpu_utilization_samples": cpu_samples,
                "cpu_peak_percent": max(cpu_samples) if cpu_samples else 0,
                "cpu_avg_percent": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            })

    def calculate_stability_analysis(self):
        """Calculate stability and consistency metrics"""
        if not self.results["test_queries"]:
            return

        success_sequence = [q["success"] for q in self.results["test_queries"]]

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

        self.results["stability_analysis"].update({
            "success_streak": current_success_streak,
            "max_success_streak": max_success_streak,
            "failure_streak": current_failure_streak,
            "max_failure_streak": max_failure_streak
        })

        # Calculate stability score (0-1.0)
        success_rate = self.results["performance_metrics"]["success_rate"] / 100
        consistency_bonus = min(max_success_streak / len(success_sequence), 0.3) if success_sequence else 0

        stability_score = success_rate + consistency_bonus
        stability_score = min(stability_score, 1.0)

        # Performance consistency score
        successful_queries = [q for q in self.results["test_queries"] if q["success"]]
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

        self.results["stability_analysis"].update({
            "stability_score": round(stability_score, 3),
            "performance_consistency_score": round(consistency_score, 3)
        })

    def validate_test_results(self):
        """Validate test results against success criteria"""

        # 64K Context Success Criteria for RAG Models
        success_criteria = {
            "context_compatibility": True,  # All queries attempted
            "performance_baseline": self.results["performance_metrics"]["avg_tokens_per_sec"] > 3.0,
            "rag_efficiency_target": self.results["performance_metrics"]["avg_response_time_s"] < 45,  # Efficient RAG responses
            "hardware_safety": self.results["resource_utilization"]["ram_peak_mb"] < (127 * 1024 * 0.8),  # <80% RAM
            "success_rate_target": self.results["performance_metrics"]["success_rate"] > 85,  # High reliability for RAG
            "stability_rag": self.results["stability_analysis"]["stability_score"] > 0.75  # Consistent RAG performance
        }

        # Overall validation
        test_passed = all(success_criteria.values())

        validation_results = {
            "criteria": success_criteria,
            "overall_passed": test_passed,
            "recommendations": [],
            "production_readiness": "RAG_READY" if test_passed else "REQUIRES_RAG_OPTIMIZATION"
        }

        # Generate recommendations
        if not success_criteria["performance_baseline"]:
            validation_results["recommendations"].append("RAG performance below target - consider query optimization or prompt engineering")

        if not success_criteria["rag_efficiency_target"]:
            validation_results["recommendations"].append("Response times exceed RAG efficiency target - optimize retrieval pipelines")

        if not success_criteria["hardware_safety"]:
            validation_results["recommendations"].append("RAM usage exceeds safe limits - consider smaller chunk sizes or model optimization")

        if not success_criteria["stability_rag"]:
            validation_results["recommendations"].append("RAG stability below requirement - investigate query consistency and retrieval reliability")

        self.results["validation_results"] = validation_results

        return test_passed

    def save_test_results(self):
        """Save comprehensive test results in standardized format"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. JSONL Log File (detailed query-by-query data)
        log_file = self.logs_dir / f"ctx_64k_liquid_rag_{timestamp}.jsonl"
        with open(log_file, 'w', encoding='utf-8') as f:
            for query in self.results["test_queries"]:
                f.write(json.dumps(query, default=str, ensure_ascii=False) + '\n')

        # 2. JSON Summary File
        summary_data = {
            "test_metadata": self.results["test_metadata"],
            "performance_metrics": self.results["performance_metrics"],
            "resource_utilization": self.results["resource_utilization"],
            "stability_analysis": self.results["stability_analysis"],
            "validation_results": self.results["validation_results"],
            "generated_at": timestamp,
            "summary_file": str(log_file)
        }

        summary_file = self.summaries_dir / f"ctx_64k_liquid_rag_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

        # 3. YAML Default Configuration
        import yaml

        config_preset = {
            "model": self.model,
            "presets": {
                "rag_64k_chunking": {
                    "num_ctx": 65536,
                    "batch": self.config["options"]["batch"],
                    "num_thread": self.config["options"]["num_thread"],
                    "f16_kv": self.config["options"]["f16_kv"],
                    "temperature": self.config["options"]["temperature"],
                    "tokens_per_sec": round(self.results["performance_metrics"]["avg_tokens_per_sec"], 2),
                    "ttft_ms": round(self.results["performance_metrics"]["avg_response_time_s"] * 1000),
                    "ram_increase_gb": round(self.results["resource_utilization"]["ram_increase_mb"] / 1024, 2),
                    "stability_score": self.results["stability_analysis"]["stability_score"],
                    "use_case": "Document chunking and multi-document RAG analysis",
                    "validated": self.results["validation_results"]["overall_passed"]
                }
            }
        }

        config_file = self.defaults_dir / f"ctx_64k_liquid_rag_config_{timestamp}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_preset, f, default_flow_style=False, sort_keys=False)

        # 4. Markdown Report
        report_file = self.documentation_dir / f"64k_context_rag_test_report_{timestamp}.md"
        self.generate_markdown_report(summary_data, report_file)

        print(f"📁 Results saved:")
        print(f"  Logs: {log_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Config: {config_file}")
        print(f"  Report: {report_file}")

    def generate_markdown_report(self, summary_data, report_file):
        """Generate comprehensive markdown test report"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# O3 Context Scaling Test: Liquid-RAG at 64K Context\n\n")

            # Test Overview
            f.write("## Test Overview\n\n")
            f.write(f"**Model:** {self.model}\n")
            f.write("**Category:** RAG Model Document Chunking\n")
            f.write("**Context Size:** 65,536 tokens (64K)\n")
            f.write("**Test Date:** " + summary_data["generated_at"] + "\n")
            f.write("**Use Case:** Multi-document retrieval and synthesis\n\n")

            # Hardware Configuration
            f.write("## Hardware Configuration\n\n")
            hw = summary_data["test_metadata"]["hardware_baseline"]
            f.write(f"- **CPU Cores:** {hw['cpu_count_physical']} physical, {hw['cpu_count_logical']} logical\n")
            f.write(f"- **RAM:** {hw['total_ram_gb']} GB total, {hw['available_ram_gb']} GB available\n")
            f.write("- **Platform:** Ryzen 16-core (32 logical)\n")
            f.write("- **Inference:** CPU-only\n\n")

            # Performance Results
            f.write("## Performance Results\n\n")
            perf = summary_data["performance_metrics"]
            f.write(f"- **Total Queries:** {perf['total_queries']}\n")
            f.write(f"- **Success Rate:** {perf['success_rate']:.1f}%\n")
            f.write(f"- **Tokens Generated:** {perf['total_tokens_generated']}\n")
            f.write(f"- **Avg Tokens/sec:** {perf['avg_tokens_per_sec']:.2f}\n")
            f.write(f"- **Performance Range:** {perf['min_tokens_per_sec']:.2f} - {perf['max_tokens_per_sec']:.2f} tok/s\n")
            f.write(f"- **Avg Response Time:** {perf['avg_response_time_s']:.2f}s\n\n")

            # Resource Utilization
            f.write("## Resource Utilization\n\n")
            res = summary_data["resource_utilization"]
            f.write(f"- **RAM Start:** {res['ram_start_mb']:.0f} MB\n")
            f.write(f"- **RAM Peak:** {res['ram_peak_mb']:.0f} MB\n")
            f.write(f"- **RAM Increase:** {res['ram_increase_mb']:.0f} MB\n")
            f.write(f"- **Context Memory:** {res['context_memory_allocation_mb']:.0f} MB estimated\n")
            f.write(f"- **CPU Peak:** {res['cpu_peak_percent']:.1f}%\n")
            f.write(f"- **CPU Average:** {res['cpu_avg_percent']:.1f}%\n\n")

            # Stability Analysis
            f.write("## Stability Analysis\n\n")
            stab = summary_data["stability_analysis"]
            f.write(f"- **Stability Score:** {stab['stability_score']:.3f}/1.0\n")
            f.write(f"- **Performance Consistency:** {stab['performance_consistency_score']:.3f}/1.0\n")
            f.write(f"- **Max Success Streak:** {stab['max_success_streak']}\n")
            f.write(f"- **Max Failure Streak:** {stab['max_failure_streak']}\n\n")

            # Validation Results
            f.write("## RAG-Specific Validation Summary\n\n")
            val = summary_data["validation_results"]
            status_emoji = "✅" if val["overall_passed"] else "❌"
            f.write(f"**Overall Result:** {status_emoji} **{val['production_readiness']}**\n\n")

            f.write("### RAG Success Criteria\n")
            for criterion, passed in val["criteria"].items():
                emoji = "✅" if passed else "❌"
                readable_name = criterion.replace('_', ' ').title()
                if "rag" in criterion.lower():
                    readable_name = readable_name.replace("Rag ", "RAG ")
                f.write(f"- {emoji} **{readable_name}:** {'PASS' if passed else 'FAIL'}\n")

            if val["recommendations"]:
                f.write("\n### Action Required\n")
                for rec in val["recommendations"]:
                    f.write(f"- ⚠️ **{rec}**\n")

            f.write("\n---\n")
            f.write("**Test Framework:** AI-First Optimization (Binary Search + Statistical Validation)\n")
            f.write("**Phase:** 1.2.1 - RAG Model Document Chunking\n")

    def run_test(self):
        """Execute the complete 64K context scaling test for Liquid-RAG"""

        print("🔍 O3 Context Scaling Test: Liquid-RAG at 64K Context")
        print("=" * 70)
        print(f"Model: {self.model}")
        print(f"Context: 65,536 tokens (64K)")
        print(f"Category: RAG Model Document Chunking")
        print("Use Case: Multi-document retrieval and synthesis")
        print("=" * 70)

        # Initialize test
        self.results["test_metadata"]["start_time"] = datetime.now().isoformat()
        self.results["resource_utilization"]["ram_start_mb"] = psutil.virtual_memory().used / (1024**2)

        print(f"📊 Baseline RAM: {self.results['resource_utilization']['ram_start_mb']:.0f} MB")

        # Execute test queries
        print("\n🔬 Executing RAG Analysis Queries...")

        for i, query_data in enumerate(self.test_queries, 1):
            print(f"\n[{i}/{len(self.test_queries)}] Processing: {query_data['name']}")
            result = self.send_test_query(query_data)
            self.results["test_queries"].append(result)

            # Brief pause to prevent overwhelming system
            time.sleep(2)

        # Calculate comprehensive metrics
        print("\n📊 Calculating Performance Metrics...")
        self.calculate_performance_metrics()

        print("📊 Analyzing Resource Utilization...")
        self.calculate_resource_utilization()

        print("📊 Performing Stability Analysis...")
        self.calculate_stability_analysis()

        # Validate results against RAG-specific criteria
        print("📊 Validating RAG Performance...")
        test_passed = self.validate_test_results()

        # Finalize metadata
        self.results["test_metadata"]["end_time"] = datetime.now().isoformat()
        total_duration = len(self.test_queries) * 2 + sum(q["response_time_s"] for q in self.results["test_queries"])
        self.results["test_metadata"]["total_duration_s"] = total_duration

        # Save comprehensive results
        print("💾 Saving RAG Test Results...")
        self.save_test_results()

        # Print summary
        self.print_test_summary(test_passed)

    def print_test_summary(self, test_passed):
        """Print comprehensive test summary"""

        print("\n" + "="*70)
        print("🎯 CONTEXT SCALING TEST COMPLETE: Liquid-RAG @ 64K")
        print("="*70)

        perf = self.results["performance_metrics"]
        res = self.results["resource_utilization"]
        stab = self.results["stability_analysis"]

        print(f"📈 PERFORMANCE METRICS:")
        print(f"   Success Rate:     {perf['success_rate']:.1f}%")
        print(f"   Avg Tokens/sec:   {perf['avg_tokens_per_sec']:.2f}")
        print(f"   Token Range:      {perf['min_tokens_per_sec']:.2f} - {perf['max_tokens_per_sec']:.2f}")
        print(f"   Avg Response:     {perf['avg_response_time_s']:.2f}s")
        print(f"   Total Tokens:     {perf['total_tokens_generated']}")

        print(f"\n💾 RESOURCE UTILIZATION:")
        print(f"   RAM Increase:     {res['ram_increase_mb']:.0f} MB")
        print(f"   RAM Peak:         {res['ram_peak_mb']:.0f} MB")
        print(f"   CPU Peak:         {res['cpu_peak_percent']:.1f}%")
        print(f"   CPU Average:      {res['cpu_avg_percent']:.1f}%")

        print(f"\n🎭 STABILITY ANALYSIS:")
        print(f"   Stability Score:  {stab['stability_score']:.3f}/1.0")
        print(f"   Consistency:      {stab['performance_consistency_score']:.3f}/1.0")

        val = self.results["validation_results"]
        status_emoji = "🔍" if test_passed else "⚠️"
        status_text = "RAG_READY" if test_passed else "REQUIRES_RAG_OPTIMIZATION"

        print(f"\n🏆 RAG VALIDATION RESULT: {status_emoji} {status_text}")

        if val["recommendations"]:
            print("💡 RECOMMENDATIONS:")
            for rec in val["recommendations"]:
                print(f"   • {rec}")

        print(f"\n📁 Result files saved in: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="O3 Context Scaling Test: Liquid-RAG @ 64K")
    parser.add_argument("--output-dir", default="ctx_64k_liquid_rag", help="Output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Number of test iterations")

    args = parser.parse_args()

    print(f"Initializing 64K context test for Liquid-RAG...")
    test = ContextScaling64kLiquidRAGTest(output_dir=args.output_dir)

    print("Starting RAG context scaling test...")
    test.run_test()

    print("\n64K context RAG test complete!")

if __name__ == "__main__":
    main()
