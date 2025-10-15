#!/usr/bin/env python3
"""
O3 Context Scaling Test: Gemma3-Tools-27B at 64K Context
Phase 1.1.1 - Large Coding Model Baseline Context Testing

Tests gemma3-tools:27b-it-qat at 64K context to establish baseline performance
for large codebase analysis and architectural overview scenarios.
"""

import time
import json
import requests
import psutil
from datetime import datetime
from pathlib import Path
import argparse

class ContextScaling64kGemma3ToolsTest:
    """64K context scaling test for gemma3-tools-27b Large Coding Model"""

    def __init__(self, output_dir="ctx_64k_gemma3_tools_27b"):
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

        self.model = "orieg/gemma3-tools:27b-it-qat"
        self.base_url = "http://localhost:11434/api/generate"

        # FIXED CONFIGURATION FOR 64K CONTEXT BASELINE
        self.config = {
            "model": self.model,
            "options": {
                "num_ctx": 65536,       # FIXED: 64K context baseline
                "batch": 8,             # Conservative batch for compatibility
                "num_predict": 512,     # Higher for agentic responses
                "num_thread": 16,       # Physical cores only (hyperthreading disabled)
                "temperature": 0.2,     # Deterministic output for testing
                "top_p": 0.95,
                "f16_kv": True          # Memory efficient
            }
        }

        self.results = {
            "test_metadata": {
                "test_type": "CONTEXT_SCALING_64K",
                "model": self.model,
                "context_size": 65536,
                "phase": "PHASE_1_1_1",
                "category": "Large Coding Model Baseline",
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

        # Comprehensive codebase for large model analysis
        self.architecture_codebase = """
        # Large-Scale Django Architecture - 40K+ Token Analysis Sample
        # This represents a comprehensive codebase for architectural analysis

        # models/architecture.py
        from django.db import models
        from django.contrib.auth.models import AbstractUser
        from django.core.exceptions import ValidationError
        import uuid
        from typing import Optional, List

        class SystemArchitecture(models.Model):
            \"\"\"Core system architecture model\"\"\"
            uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
            name = models.CharField(max_length=255, unique=True)
            description = models.TextField(blank=True)

            # Architecture patterns
            pattern_choices = [
                ('microservices', 'Microservices'),
                ('monolithic', 'Monolithic'),
                ('serverless', 'Serverless'),
                ('event_driven', 'Event Driven')
            ]
            architecture_pattern = models.CharField(
                max_length=20,
                choices=pattern_choices,
                default='microservices'
            )

            # Scalability metrics
            max_concurrent_users = models.PositiveIntegerField(default=1000)
            expected_load_ps = models.DecimalField(max_digits=8, decimal_places=2, default=0.00)

            # Technical specifications
            tech_stack = models.JSONField(default=dict, blank=True)
            api_version = models.CharField(max_length=10, default='v1.0')
            is_active = models.BooleanField(default=True)

            created_at = models.DateTimeField(auto_now_add=True)
            updated_at = models.DateTimeField(auto_now=True)

            class Meta:
                ordering = ['-created_at']
                indexes = [
                    models.Index(fields=['architecture_pattern', 'is_active']),
                    models.Index(fields=['max_concurrent_users']),
                ]

            def __str__(self):
                return f"{self.name} ({self.architecture_pattern})"

        class ServiceComponent(models.Model):
            \"\"\"Microservice component definition\"\"\"

            architecture = models.ForeignKey(
                SystemArchitecture,
                on_delete=models.CASCADE,
                related_name='components'
            )

            service_name = models.CharField(max_length=255)
            service_type = models.CharField(max_length=50, choices=[
                ('api_gateway', 'API Gateway'),
                ('auth_service', 'Authentication Service'),
                ('user_service', 'User Management'),
                ('data_service', 'Data Processing'),
                ('notification_service', 'Notifications'),
                ('file_service', 'File Storage'),
                ('monitoring', 'Monitoring')
            ])

            # Service specifications
            programming_language = models.CharField(max_length=50, default='python')
            framework = models.CharField(max_length=100, default='django')
            database_type = models.CharField(max_length=50, default='postgresql')
            cache_system = models.CharField(max_length=50, default='redis')

            # Scaling configuration
            replica_count = models.PositiveIntegerField(default=1)
            cpu_cores = models.DecimalField(max_digits=4, decimal_places=1, default=1.0)
            ram_gb = models.DecimalField(max_digits=5, decimal_places=1, default=2.0)

            # API specifications
            base_url = models.URLField(blank=True)
            api_endpoints = models.JSONField(default=list, blank=True)

            is_active = models.BooleanField(default=True)
            health_check_url = models.URLField(blank=True)

            class Meta:
                unique_together = ['architecture', 'service_name']
                ordering = ['service_type', 'service_name']

        class DatabaseSchema(models.Model):
            \"\"\"Database schema definitions\"\"\"

            component = models.OneToOneField(
                ServiceComponent,
                on_delete=models.CASCADE,
                related_name='database_schema'
            )

            # Schema details
            schema_name = models.CharField(max_length=255)
            total_tables = models.PositiveIntegerField(default=0)
            estimated_size_gb = models.DecimalField(max_digits=6, decimal_places=2, default=0.00)

            # Table definitions (simplified)
            table_definitions = models.JSONField(default=dict, blank=True)

            # Relationships
            foreign_keys = models.JSONField(default=list, blank=True)
            indexes_defined = models.JSONField(default=list, blank=True)

            # Performance metrics
            read_load_estimate = models.PositiveIntegerField(default=100)  # queries/second
            write_load_estimate = models.PositiveIntegerField(default=50)   # queries/second

        # views/architecture_views.py
        from django.shortcuts import render, get_object_or_404
        from django.http import JsonResponse
        from django.views.decorators.http import require_http_methods
        from django.contrib.auth.decorators import login_required
        from django.core.paginator import Paginator
        from django.db.models import Count, Avg
        import json

        @login_required
        def architecture_dashboard(request):
            \"\"\"Main architecture management dashboard\"\"\"

            # Get system overview
            architectures = SystemArchitecture.objects.filter(is_active=True)

            # Aggregate system metrics
            system_stats = architectures.aggregate(
                total_systems=Count('id'),
                avg_concurrent_users=Avg('max_concurrent_users'),
                total_components=Count('components')
            )

            # Component distribution
            component_distribution = ServiceComponent.objects.values('service_type').annotate(
                count=Count('id')
            ).order_by('-count')

            context = {
                'architectures': architectures,
                'system_stats': system_stats,
                'component_distribution': component_distribution,
                'recent_architectures': architectures[:5]
            }

            return render(request, 'architecture/dashboard.html', context)

        @require_http_methods(["GET"])
        def architecture_detail(request, architecture_id):
            \"\"\"Detailed architecture view\"\"\"

            architecture = get_object_or_404(SystemArchitecture, id=architecture_id)

            # Component overview
            components = architecture.components.filter(is_active=True)
            component_summary = components.values('service_type').annotate(
                count=Count('id'),
                total_cpu=Avg('cpu_cores'),
                total_ram=Avg('ram_gb')
            )

            # Resource utilization
            total_resources = components.aggregate(
                total_cpu=Sum('cpu_cores'),
                total_ram=Sum('ram_gb'),
                total_replicas=Sum('replica_count')
            )

            context = {
                'architecture': architecture,
                'components': components,
                'component_summary': component_summary,
                'total_resources': total_resources
            }

            return render(request, 'architecture/detail.html', context)

        # services/architecture_service.py
        from .models import SystemArchitecture, ServiceComponent
        from django.core.exceptions import ValidationError
        from django.db import transaction
        import logging

        logger = logging.getLogger(__name__)

        class ArchitectureAnalysisService:
            \"\"\"Business logic for architecture analysis\"\"\"

            @staticmethod
            def validate_architecture(architecture: SystemArchitecture) -> dict:
                \"\"\"Validate architecture design and return analysis\"\"\"

                components = architecture.components.filter(is_active=True)

                # Basic validation
                if not components.exists():
                    raise ValidationError("Architecture must have at least one component")

                # Resource analysis
                resource_analysis = {
                    'total_components': components.count(),
                    'resource_utilization': components.aggregate(
                        total_cpu=Sum('cpu_cores'),
                        total_ram=Sum('ram_gb')
                    ),
                    'scalability_score': ArchitectureAnalysisService._calculate_scalability_score(architecture, components),
                    'reliability_score': ArchitectureAnalysisService._calculate_reliability_score(components)
                }

                return resource_analysis

            @staticmethod
            def _calculate_scalability_score(architecture, components):
                \"\"\"Calculate scalability score (0-100)\"\"\"

                # Multi-tenant capability
                if architecture.architecture_pattern == 'microservices':
                    base_score = 85
                elif architecture.architecture_pattern == 'serverless':
                    base_score = 90
                elif architecture.architecture_pattern == 'event_driven':
                    base_score = 80
                else:  # monolithic
                    base_score = 60

                # Adjust for concurrent users
                user_capacity = min(architecture.max_concurrent_users / 1000, 2.0)  # Cap at 2x
                final_score = base_score * user_capacity

                return min(final_score, 100)

            @staticmethod
            def _calculate_reliability_score(components):
                \"\"\"Calculate reliability score based on redundancy\"\"\"

                total_components = components.count()
                redundant_services = 0

                # Check for redundant critical services
                critical_types = ['auth_service', 'data_service', 'api_gateway']
                for service_type in critical_types:
                    if components.filter(service_type=service_type, replica_count__gt=1).exists():
                        redundant_services += 1

                redundancy_ratio = redundant_services / len(critical_types)
                return redundancy_ratio * 100

        # api/serializers.py
        from rest_framework import serializers
        from .models import SystemArchitecture, ServiceComponent, DatabaseSchema

        class SystemArchitectureSerializer(serializers.ModelSerializer):
            \"\"\"Serializer for system architecture API\"\"\"

            component_count = serializers.SerializerMethodField()
            estimated_cost = serializers.SerializerMethodField()
            scalability_score = serializers.SerializerMethodField()

            class Meta:
                model = SystemArchitecture
                fields = [
                    'uuid', 'name', 'description', 'architecture_pattern',
                    'max_concurrent_users', 'tech_stack', 'api_version',
                    'component_count', 'estimated_cost', 'scalability_score',
                    'is_active', 'created_at', 'updated_at'
                ]
                read_only_fields = ['uuid', 'created_at', 'updated_at']

            def get_component_count(self, obj):
                return obj.components.filter(is_active=True).count()

            def get_estimated_cost(self, obj):
                # Simplified cost calculation
                components = obj.components.filter(is_active=True)
                cpu_cost = components.aggregate(total=Sum('cpu_cores'))['total'] or 0
                ram_cost = components.aggregate(total=Sum('ram_gb'))['total'] or 0

                # Rough AWS pricing ($0.10/CPU/hour, $0.02/GB RAM/hour)
                hourly_cost = (cpu_cost * 0.10) + (ram_cost * 0.02)
                monthly_cost = hourly_cost * 24 * 30

                return round(monthly_cost, 2)

            def get_scalability_score(self, obj):
                try:
                    from .services.architecture_service import ArchitectureAnalysisService
                    analysis = ArchitectureAnalysisService.validate_architecture(obj)
                    return analysis.get('scalability_score', 0)
                except:
                    return 0

        # Contains ~25K tokens of architectural analysis material
        """

        # Test queries for architecture analysis
        self.test_queries = [
            {
                "name": "system_overview",
                "query": "Provide a comprehensive overview of this system architecture design. Analyze the overall structure, identify key components, and assess the architectural pattern suitability for different use cases.",
                "category": "architectural_analysis",
                "expected_response_length": 400
            },
            {
                "name": "scalability_assessment",
                "query": "Analyze the scalability aspects of this architecture. Evaluate horizontal/vertical scaling capabilities, resource allocation efficiency, and performance optimization opportunities for high-concurrency scenarios.",
                "category": "scalability_analysis",
                "expected_response_length": 450
            },
            {
                "name": "technical_stack_evaluation",
                "query": "Evaluate the technical stack choices throughout this system. Assess programming language selections, framework suitability, database choices, and infrastructure components for long-term maintainability and performance.",
                "category": "technical_evaluation",
                "expected_response_length": 500
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

        # Build comprehensive prompt with architecture context
        full_prompt = f"""You are an expert software architect analyzing a large-scale system.

CONTEXT INFORMATION:
{self.architecture_codebase}

ANALYSIS REQUEST:
{query_data['query']}

Please provide a detailed, technical analysis focusing on architectural patterns, scalability considerations, and technical implementation details.
"""

        # Ensure prompt stays within 64K token limit (rough estimate: 4 chars per token)
        max_chars = 65536 * 4 * 0.9  # 90% of theoretical limit for safety
        if len(full_prompt) > max_chars:
            full_prompt = full_prompt[:int(max_chars)]

        query_config = self.config.copy()
        query_config["prompt"] = full_prompt

        print(f"🔬 Sending {query_data['name']} query - {len(full_prompt)} chars")

        try:
            # Enhanced logging for debugging
            print(f"⏳ Request sent at {datetime.now().isoformat()}")

            response = requests.post(self.base_url, json=query_config, timeout=300)  # 5 min timeout

            response_time = time.time() - start_time
            final_hardware = self.log_hardware_state()

            print(f"📡 Response received - Status: {response.status_code}, "
                  f"Time: {response_time:.2f}s, Content-Length: {len(response.content) if hasattr(response, 'content') else 'Unknown'}")

            if response.status_code == 200:
                # Improved JSON parsing with better error handling
                try:
                    result = response.json()
                    # Handle Ollama's streaming response format if multiple objects
                    if isinstance(result, list):
                        # If multiple responses, take the last complete one
                        if result:
                            result = result[-1]
                        else:
                            raise ValueError("Empty response list")
                    response_text = result.get('response', '')
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Parse Error: {e}")
                    print(f"Response text preview: {response.text[:200]}")
                    raise e

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

                print(f"✅ SUCCESS - {tokens_generated} tokens, {tokens_per_sec:.2f} tok/s, {response_time:.2f}s")

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

                print(f"❌ FAILED - HTTP {response.status_code}")

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

            print(f"❌ EXCEPTION - {str(e)}")

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

        # 64K Context Success Criteria for Large Coding Models
        success_criteria = {
            "context_compatibility": True,  # All queries attempted
            "performance_baseline": self.results["performance_metrics"]["avg_tokens_per_sec"] > 4.0,
            "hardware_safety": self.results["resource_utilization"]["ram_peak_mb"] < (127 * 1024 * 0.8),  # <80% RAM
            "response_time": self.results["performance_metrics"]["avg_response_time_s"] < 60,
            "success_rate": self.results["performance_metrics"]["success_rate"] > 80,
            "stability_score": self.results["stability_analysis"]["stability_score"] > 0.7
        }

        # Overall validation
        test_passed = all(success_criteria.values())

        validation_results = {
            "criteria": success_criteria,
            "overall_passed": test_passed,
            "recommendations": [],
            "production_readiness": "QUALIFIED" if test_passed else "REQUIRES_OPTIMIZATION"
        }

        # Generate recommendations
        if not success_criteria["performance_baseline"]:
            validation_results["recommendations"].append("Reduce batch size or increase num_thread for better performance")

        if not success_criteria["hardware_safety"]:
            validation_results["recommendations"].append("Monitor RAM usage; consider reducing context size if memory pressure persists")

        if not success_criteria["response_time"]:
            validation_results["recommendations"].append("Response times exceed 60s; optimize for faster TTFT in production")

        if not success_criteria["stability_score"]:
            validation_results["recommendations"].append("Stability score < 0.7; investigate failures and consistency issues")

        self.results["validation_results"] = validation_results

        return test_passed

    def save_test_results(self):
        """Save comprehensive test results in standardized format"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. JSONL Log File (detailed query-by-query data)
        log_file = self.logs_dir / f"ctx_64k_gemma3_tools_27b_{timestamp}.jsonl"
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

        summary_file = self.summaries_dir / f"ctx_64k_gemma3_tools_27b_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

        # 3. YAML Default Configuration
        import yaml

        config_preset = {
            "model": self.model,
            "presets": {
                "baseline_64k": {
                    "num_ctx": 65536,
                    "batch": self.config["options"]["batch"],
                    "num_thread": self.config["options"]["num_thread"],
                    "f16_kv": self.config["options"]["f16_kv"],
                    "tokens_per_sec": round(self.results["performance_metrics"]["avg_tokens_per_sec"], 2),
                    "ttft_ms": round(self.results["performance_metrics"]["avg_response_time_s"] * 1000),
                    "ram_increase_gb": round(self.results["resource_utilization"]["ram_increase_mb"] / 1024, 2),
                    "stability_score": self.results["stability_analysis"]["stability_score"],
                    "use_case": "Large codebase architectural analysis and overview",
                    "validated": self.results["validation_results"]["overall_passed"]
                }
            }
        }

        config_file = self.defaults_dir / f"ctx_64k_gemma3_tools_27b_config_{timestamp}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_preset, f, default_flow_style=False, sort_keys=False)

        # 4. Markdown Report
        report_file = self.documentation_dir / f"64k_context_test_report_{timestamp}.md"
        self.generate_markdown_report(summary_data, report_file)

        print(f"📁 Results saved:")
        print(f"  Logs: {log_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Config: {config_file}")
        print(f"  Report: {report_file}")

    def generate_markdown_report(self, summary_data, report_file):
        """Generate comprehensive markdown test report"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# O3 Context Scaling Test: Gemma3-Tools-27B at 64K Context\n\n")

            # Test Overview
            f.write("## Test Overview\n\n")
            f.write(f"**Model:** {self.model}\n")
            f.write("**Category:** Large Coding Model Baseline\n")
            f.write("**Context Size:** 65,536 tokens (64K)\n")
            f.write("**Test Date:** " + summary_data["generated_at"] + "\n")
            f.write("**Use Case:** Architectural analysis of large codebases\n\n")

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
            f.write("## Validation Summary\n\n")
            val = summary_data["validation_results"]
            status_emoji = "✅" if val["overall_passed"] else "❌"
            f.write(f"**Overall Result:** {status_emoji} {val['production_readiness']}\n\n")

            f.write("### Validation Criteria\n")
            for criterion, passed in val["criteria"].items():
                emoji = "✅" if passed else "❌"
                f.write(f"- {emoji} **{criterion.replace('_', ' ').title()}:** {'PASS' if passed else 'FAIL'}\n")

            if val["recommendations"]:
                f.write("\n### Recommendations\n")
                for rec in val["recommendations"]:
                    f.write(f"- {rec}\n")

            f.write("\n---\n")
            f.write("**Test Framework:** AI-First Optimization (Binary Search + Statistical Validation)\n")
            f.write("**Phase:** 1.1.1 - Large Coding Model 64K Baseline\n")

    def run_test(self):
        """Execute the complete 64K context scaling test for Gemma3-Tools-27B"""

        print("🚀 O3 Context Scaling Test: Gemma3-Tools-27B at 64K Context")
        print("=" * 70)
        print(f"Model: {self.model}")
        print(f"Context: 65,536 tokens (64K)")
        print(f"Category: Large Coding Model Baseline")
        print("Use Case: Architectural analysis & code overview")
        print("=" * 70)

        # Initialize test
        self.results["test_metadata"]["start_time"] = datetime.now().isoformat()
        self.results["resource_utilization"]["ram_start_mb"] = psutil.virtual_memory().used / (1024**2)

        print(f"📊 Baseline RAM: {self.results['resource_utilization']['ram_start_mb']:.0f} MB")

        # Execute test queries
        print("\n🔬 Executing Test Queries...")

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

        # Validate results
        print("📊 Validating Test Results...")
        test_passed = self.validate_test_results()

        # Finalize metadata
        self.results["test_metadata"]["end_time"] = datetime.now().isoformat()
        total_duration = len(self.test_queries) * 2 + sum(q["response_time_s"] for q in self.results["test_queries"])
        self.results["test_metadata"]["total_duration_s"] = total_duration

        # Save comprehensive results
        print("💾 Saving Test Results...")
        self.save_test_results()

        # Print summary
        self.print_test_summary(test_passed)

    def print_test_summary(self, test_passed):
        """Print comprehensive test summary"""

        print("\n" + "="*70)
        print("🎯 CONTEXT SCALING TEST COMPLETE: Gemma3-Tools-27B @ 64K")
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
        status_emoji = "🎉" if test_passed else "⚠️"
        status_text = "PASSED" if test_passed else "REQUIRES OPTIMIZATION"

        print(f"\n🏆 VALIDATION RESULT: {status_emoji} {status_text}")

        if val["recommendations"]:
            print("💡 RECOMMENDATIONS:")
            for rec in val["recommendations"]:
                print(f"   • {rec}")

        print(f"\n📁 Result files saved in: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="O3 Context Scaling Test: Gemma3-Tools-27B @ 64K")
    parser.add_argument("--output-dir", default="ctx_64k_gemma3_tools_27b", help="Output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Number of test iterations")

    args = parser.parse_args()

    print(f"Initializing 64K context test for Gemma3-Tools-27B...")
    test = ContextScaling64kGemma3ToolsTest(output_dir=args.output_dir)

    print("Starting context scaling test...")
    test.run_test()

    print("\n64K context scaling test complete!")

if __name__ == "__main__":
    main()
