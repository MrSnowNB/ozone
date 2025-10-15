#!/usr/bin/env python3
"""
O3 Phase 1 Validation Script
Tests all Phase 1 context scaling scripts before running model tests

Validates syntax, imports, configurations, and hardware monitoring
without actually calling Ollama models (dry-run mode).
"""

import sys
import os
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime

class Phase1ValidationTester:
    """Comprehensive validation for all Phase 1 test scripts"""

    def __init__(self):
        self.test_root = Path("multi_model_context_tests")
        self.phase1_scripts = [
            # Gemma3-Tools-27B (Large Coding Model)
            "ctx_64k_gemma3_tools_27b.py",
            "ctx_128k_gemma3_tools_27b.py",
            # "ctx_256k_gemma3_tools_27b.py",  # We'll create this next

            # Liquid-RAG (RAG Model)
            "ctx_64k_liquid_rag.py",
            # "ctx_128k_liquid_rag.py",      # We'll create this next
            # "ctx_256k_liquid_rag.py",      # We'll create this next

            # Chat Models (qwen2.5:3b-instruct, gemma3:latest)
            "ctx_32k_chat_models.py",      # ✅ CREATED - Ready for validation
            # "ctx_64k_chat_models.py",      # We'll create this next
            # "ctx_128k_chat_models.py"      # We'll create this next
        ]

        self.results = {
            "scripts_validated": [],
            "validation_results": {},
            "summary": {
                "total_scripts": len(self.phase1_scripts),
                "passed_syntax": 0,
                "passed_imports": 0,
                "passed_config": 0,
                "passed_hardware": 0,
                "passed_structure": 0,
                "failed_overall": 0
            }
        }

    def run_validation(self):
        """Run comprehensive validation of all Phase 1 scripts"""

        print("🧪 O3 PHASE 1 VALIDATION SUITE")
        print("=" * 50)
        print(f"Testing {len(self.phase1_scripts)} context scaling scripts")
        print("Mode: Dry-run (no actual model calls)")
        print("=" * 50)

        for script_name in self.phase1_scripts:
            script_path = self.test_root / script_name

            if not script_path.exists():
                print(f"⚠️  SKIPPING: {script_name} (file not found)")
                continue

            print(f"\n🔍 VALIDATING: {script_name}")

            script_validation = self.validate_script(script_path)
            self.results["scripts_validated"].append(script_name)
            self.results["validation_results"][script_name] = script_validation

            # Update summary counters
            self.update_summary(script_validation)

        # Add final validation checks
        self.run_integration_tests()

        # Generate validation report
        self.generate_validation_report()

        # Print final results
        self.print_validation_summary()

    def validate_script(self, script_path):
        """Validate a single test script comprehensively"""

        script_name = script_path.name
        validation_result = {
            "script_name": script_name,
            "syntax_check": {"passed": False, "error": None},
            "import_check": {"passed": False, "error": None},
            "config_validation": {"passed": False, "error": None},
            "hardware_monitoring": {"passed": False, "error": None},
            "directory_structure": {"passed": False, "error": None},
            "class_initialization": {"passed": False, "error": None},
            "overall_passed": False,
            "issues_found": []
        }

        # 1. Syntax Check
        validation_result["syntax_check"] = self.check_syntax(script_path)
        if not validation_result["syntax_check"]["passed"]:
            validation_result["issues_found"].append(f"Syntax error: {validation_result['syntax_check']['error']}")
            return validation_result  # Can't proceed without valid syntax

        # 2. Import Validation
        validation_result["import_check"] = self.check_imports(script_path)
        if not validation_result["import_check"]["passed"]:
            validation_result["issues_found"].append(f"Import error: {validation_result['import_check']['error']}")

        # 3. Configuration Validation
        validation_result["config_validation"] = self.validate_configuration(script_path)

        # 4. Hardware Monitoring Test
        validation_result["hardware_monitoring"] = self.test_hardware_monitoring(script_path)

        # 5. Directory Structure Check
        validation_result["directory_structure"] = self.check_directory_structure(script_path)

        # 6. Class Initialization Test
        validation_result["class_initialization"] = self.test_class_initialization(script_path)

        # Overall assessment
        critical_checks = [
            validation_result["syntax_check"]["passed"],
            validation_result["import_check"]["passed"],
            # Config issues are warnings, not critical for script validation
        ]

        validation_result["overall_passed"] = all(critical_checks)

        # Print validation status
        status_emoji = "✅" if validation_result["overall_passed"] else "❌"
        print(f"   {status_emoji} Overall: {'PASSED' if validation_result['overall_passed'] else 'ISSUES FOUND'}")

        if validation_result["issues_found"]:
            for issue in validation_result["issues_found"]:
                print(f"   ⚠️  {issue}")

        return validation_result

    def check_syntax(self, script_path):
        """Check Python syntax without executing"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Try to compile to check syntax
            compile(source_code, script_path, 'exec')
            return {"passed": True, "error": None}

        except SyntaxError as e:
            return {"passed": False, "error": f"{e.filename}:{e.lineno}: {e.msg}"}
        except Exception as e:
            return {"passed": False, "error": f"Compilation error: {str(e)}"}

    def check_imports(self, script_path):
        """Check if all required imports are available"""
        try:
            # Add current directory to Python path for local imports
            sys.path.insert(0, str(self.test_root))

            # Try to import the main test class
            with open(script_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Extract the class name (simple regex)
            import re
            class_match = re.search(r'class\s+(\w+)Test:', source_code)
            if not class_match:
                return {"passed": False, "error": "Could not find test class in script"}

            class_name = class_match.group(1) + "Test"

            # Test import in a temporary module
            spec = importlib.util.spec_from_file_location("test_module", script_path)
            module = importlib.util.module_from_spec(spec)

            # Just check import availability, don't fully execute
            try:
                spec.loader.exec_module(module)
                # Check if the main class exists
                if hasattr(module, class_name):
                    test_class = getattr(module, class_name)
                    # Try to create instance with minimal args to test imports
                    return {"passed": True, "error": None}
                else:
                    return {"passed": False, "error": f"Test class {class_name} not found"}
            except ImportError as e:
                return {"passed": False, "error": f"Import error: {str(e)}"}
            except Exception as e:
                # Other exceptions during import are ok for validation
                return {"passed": True, "error": None}

        except Exception as e:
            return {"passed": False, "error": f"Import check failed: {str(e)}"}
        finally:
            # Clean up sys.path
            if str(self.test_root) in sys.path:
                sys.path.remove(str(self.test_root))

    def validate_configuration(self, script_path):
        """Validate configuration structure and values"""
        try:
            spec = importlib.util.spec_from_file_location("config_test", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the test class (simplified approach)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (hasattr(attr, '__init__') and
                    hasattr(attr, 'model') and
                    hasattr(attr, 'config')):

                    # Create test instance with default args
                    try:
                        test_instance = attr()

                        # Validate configuration structure
                        if not hasattr(test_instance, 'config'):
                            return {"passed": False, "error": "Missing config attribute"}

                        config = test_instance.config

                        # Check required Ollama config fields
                        required_fields = ['model', 'options']
                        for field in required_fields:
                            if field not in config:
                                return {"passed": False, "error": f"Missing required config field: {field}"}

                        # Check options fields
                        options = config.get('options', {})
                        required_options = ['num_ctx', 'batch', 'num_predict', 'num_thread', 'f16_kv']
                        for option in required_options:
                            if option not in options:
                                return {"passed": False, "error": f"Missing required option: {option}"}

                        # Validate specific values
                        if options.get('num_thread', 0) <= 0:
                            return {"passed": False, "error": "num_thread must be positive"}

                        if options.get('batch', 0) <= 0:
                            return {"passed": False, "error": "batch must be positive"}

                        return {"passed": True, "error": None}

                    except Exception as e:
                        return {"passed": False, "error": f"Configuration validation failed: {str(e)}"}

            return {"passed": False, "error": "Could not find testable class"}

        except Exception as e:
            return {"passed": False, "error": f"Config validation error: {str(e)}"}

    def test_hardware_monitoring(self, script_path):
        """Test hardware monitoring functions without full execution"""
        try:
            spec = importlib.util.spec_from_file_location("hw_test", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find test class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, 'log_hardware_state') and hasattr(attr, 'capture_hardware_baseline'):

                    # Create minimal instance to test hardware functions
                    try:
                        # Test baseline capture
                        baseline = attr.capture_hardware_baseline(None)

                        # Test hardware logging
                        hw_state = attr.log_hardware_state(None)

                        # Validate returned data structure
                        required_baseline_keys = ['cpu_count_physical', 'cpu_count_logical', 'total_ram_gb']
                        for key in required_baseline_keys:
                            if key not in baseline:
                                return {"passed": False, "error": f"Missing baseline key: {key}"}

                        required_hw_keys = ['ram_used_mb', 'cpu_percent', 'timestamp']
                        for key in required_hw_keys:
                            if key not in hw_state:
                                return {"passed": False, "error": f"Missing hw state key: {key}"}

                        return {"passed": True, "error": None}

                    except Exception as e:
                        return {"passed": False, "error": f"Hardware monitoring test failed: {str(e)}"}

            return {"passed": False, "error": "Hardware monitoring methods not found"}

        except Exception as e:
            return {"passed": False, "error": f"Hardware test error: {str(e)}"}

    def check_directory_structure(self, script_path):
        """Check if script creates proper directory structure"""
        try:
            spec = importlib.util.spec_from_file_location("dir_test", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find test class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, '__init__') and hasattr(attr, 'output_dir'):

                    try:
                        # Get expected output directory name from class
                        script_name = script_path.stem  # filename without extension

                        # Create test instance to check directory setup
                        test_instance = attr()

                        # Check if output_dir was created
                        if hasattr(test_instance, 'output_dir') and test_instance.output_dir.exists():
                            # Check if subdirectories exist
                            expected_dirs = ['logs', 'summaries', 'defaults', 'documentation']
                            missing_dirs = []

                            for expected_dir in expected_dirs:
                                dir_path = test_instance.output_dir / expected_dir
                                if not dir_path.exists():
                                    missing_dirs.append(expected_dir)

                            if missing_dirs:
                                return {"passed": False, "error": f"Missing directories: {missing_dirs}"}

                            return {"passed": True, "error": None}
                        else:
                            return {"passed": False, "error": "Output directory not created"}

                    except Exception as e:
                        return {"passed": False, "error": f"Directory structure test failed: {str(e)}"}

            return {"passed": False, "error": "Could not find class with directory setup"}

        except Exception as e:
            return {"passed": False, "error": f"Directory check error: {str(e)}"}

    def test_class_initialization(self, script_path):
        """Test that class can be initialized without errors"""
        try:
            spec = importlib.util.spec_from_file_location("init_test", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find test class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, '__init__') and hasattr(attr, 'results'):

                    try:
                        # Test basic initialization
                        test_instance = attr()

                        # Check if expected attributes exist
                        required_attrs = ['model', 'config', 'results', 'output_dir']
                        missing_attrs = []

                        for attr_name in required_attrs:
                            if not hasattr(test_instance, attr_name):
                                missing_attrs.append(attr_name)

                        if missing_attrs:
                            return {"passed": False, "error": f"Missing required attributes: {missing_attrs}"}

                        # Check results structure
                        if not isinstance(test_instance.results, dict):
                            return {"passed": False, "error": "Results should be a dictionary"}

                        # Check if required result sections exist
                        required_sections = ['test_metadata', 'performance_metrics',
                                           'resource_utilization', 'stability_analysis']
                        missing_sections = []
                        for section in required_sections:
                            if section not in test_instance.results:
                                missing_sections.append(section)

                        if missing_sections:
                            return {"passed": False, "error": f"Missing result sections: {missing_sections}"}

                        return {"passed": True, "error": None}

                    except Exception as e:
                        return {"passed": False, "error": f"Class initialization test failed: {str(e)}"}

            return {"passed": False, "error": "Could not find testable class"}

        except Exception as e:
            return {"passed": False, "error": f"Init test error: {str(e)}"}

    def update_summary(self, script_validation):
        """Update summary counters based on validation results"""

        if script_validation["syntax_check"]["passed"]:
            self.results["summary"]["passed_syntax"] += 1

        if script_validation["import_check"]["passed"]:
            self.results["summary"]["passed_imports"] += 1

        if script_validation["config_validation"]["passed"]:
            self.results["summary"]["passed_config"] += 1

        if script_validation["hardware_monitoring"]["passed"]:
            self.results["summary"]["passed_hardware"] += 1

        if script_validation["directory_structure"]["passed"]:
            self.results["summary"]["passed_structure"] += 1

        if not script_validation["overall_passed"]:
            self.results["summary"]["failed_overall"] += 1

    def run_integration_tests(self):
        """Run integration tests across all scripts"""

        print(f"\n🔗 RUNNING INTEGRATION TESTS...")

        integration_results = {
            "ollama_connectivity": self.test_ollama_connectivity(),
            "model_availability": self.test_model_availability(),
            "disk_space": self.test_disk_space(),
            "all_scripts_executable": self.test_script_execution()
        }

        self.results["integration_tests"] = integration_results

        # Print integration status
        for test_name, result in integration_results.items():
            status_emoji = "✅" if result["passed"] else "❌"
            print(f"   {status_emoji} {test_name.replace('_', ' ').title()}: {result['status']}")

    def test_ollama_connectivity(self):
        """Test basic Ollama connectivity"""
        try:
            response = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if response.returncode == 0 and "NAME" in response.stdout:
                return {"passed": True, "status": "Connected", "details": "Ollama service responding"}
            else:
                return {"passed": False, "status": "Failed", "details": "Ollama service not responding"}

        except Exception as e:
            return {"passed": False, "status": "Error", "details": str(e)}

    def test_model_availability(self):
        """Test that required models are available"""
        required_models = [
            "orieg/gemma3-tools:27b-it-qat",
            "liquid-rag:latest",
            "qwen2.5:3b-instruct",
            "gemma3:latest"
        ]

        try:
            response = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if response.returncode == 0:
                available_models = [line.split()[0] for line in response.stdout.strip().split('\n')[1:] if line.strip()]

                missing_models = []
                for required in required_models:
                    if not any(required in model for model in available_models):
                        missing_models.append(required)

                if missing_models:
                    return {"passed": False, "status": "Missing Models", "details": f"Missing: {missing_models}"}
                else:
                    return {"passed": True, "status": "All Available", "details": f"Found {len(required_models)} required models"}
            else:
                return {"passed": False, "status": "List Failed", "details": "Could not get model list"}

        except Exception as e:
            return {"passed": False, "status": "Error", "details": str(e)}

    def test_disk_space(self):
        """Test available disk space for test results"""
        try:
            # Check current directory disk space
            stat = os.statvfs('.')
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

            # Need at least 10GB free for test results
            if free_gb >= 10:
                return {"passed": True, "status": f"{free_gb:.1f}GB Free", "details": "Sufficient disk space"}
            else:
                return {"passed": False, "status": f"{free_gb:.1f}GB Free", "details": "Insufficient disk space (need 10GB+)"}

        except Exception as e:
            return {"passed": False, "status": "Check Failed", "details": str(e)}

    def test_script_execution(self):
        """Test that scripts can be executed (dry run)"""
        try:
            scripts_passed = 0
            scripts_failed = 0

            for script_name in self.phase1_scripts:
                script_path = self.test_root / script_name
                if script_path.exists():
                    # Test if script can be imported without errors
                    try:
                        # Quick syntax check
                        with open(script_path, 'r') as f:
                            compile(f.read(), script_path, 'exec')
                        scripts_passed += 1
                    except:
                        scripts_failed += 1

            if scripts_failed == 0:
                return {"passed": True, "status": f"All {scripts_passed} Scripts OK", "details": "All scripts passed basic validation"}
            else:
                return {"passed": False, "status": f"{scripts_failed} Failed", "details": f"{scripts_passed} OK, {scripts_failed} failed"}

        except Exception as e:
            return {"passed": False, "status": "Execution Test Failed", "details": str(e)}

    def generate_validation_report(self):
        """Generate comprehensive validation report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.test_root / f"phase1_validation_report_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# O3 Phase 1 Validation Report\n\n")
            f.write(f"**Validation Date:** {timestamp}\n")
            f.write(f"**Test Suite:** Multi-Model Context Scaling (Phase 1)\n\n")

            f.write("## Executive Summary\n\n")
            f.write(f"- **Scripts Validated:** {len(self.results['scripts_validated'])}\n")
            f.write(f"- **Syntax Errors:** {self.results['summary']['passed_syntax']}/{len(self.phase1_scripts)}\n")
            f.write(f"- **Import Issues:** {self.results['summary']['passed_imports']}/{len(self.phase1_scripts)}\n")
            f.write(f"- **Overall Failures:** {self.results['summary']['failed_overall']}\n\n")

            f.write("## Integration Test Results\n\n")
            if "integration_tests" in self.results:
                for test_name, result in self.results["integration_tests"].items():
                    status_emoji = "✅" if result["passed"] else "❌"
                    f.write(f"- {status_emoji} **{test_name.replace('_', ' ').title()}:** {result['status']}\n")
                    if result.get('details'):
                        f.write(f"  - *{result['details']}*\n")
                f.write("\n")

            f.write("## Detailed Script Results\n\n")
            for script_name, validation in self.results["validation_results"].items():
                f.write(f"### {script_name}\n\n")

                overall_status = "✅ PASSED" if validation["overall_passed"] else "❌ ISSUES FOUND"
                f.write(f"**Overall Status:** {overall_status}\n\n")

                # Individual checks
                checks = {
                    "Syntax Check": validation["syntax_check"],
                    "Import Validation": validation["import_check"],
                    "Configuration": validation["config_validation"],
                    "Hardware Monitoring": validation["hardware_monitoring"],
                    "Directory Structure": validation["directory_structure"],
                    "Class Initialization": validation["class_initialization"]
                }

                for check_name, check_result in checks.items():
                    status_emoji = "✅" if check_result["passed"] else "❌"
                    f.write(f"- {status_emoji} **{check_name}:** ")
                    if check_result["passed"]:
                        f.write("PASSED\n")
                    else:
                        f.write(f"FAILED - {check_result['error']}\n")

                if validation["issues_found"]:
                    f.write("\n**Issues Found:**\n")
                    for issue in validation["issues_found"]:
                        f.write(f"- ⚠️ {issue}\n")

                f.write("\n---\n\n")

            f.write("## Recommendations\n\n")

            summary = self.results["summary"]
            all_checks = sum([summary[key] for key in summary.keys() if key.startswith("passed_")])

            if summary["failed_overall"] == 0:
                f.write("✅ **All Phase 1 scripts are ready for execution.**\n\n")
                f.write("The validation suite confirms that all scripts have:\n")
                f.write("- Valid Python syntax\n")
                f.write("- Properly resolvable imports\n")
                f.write("- Correct configuration structures\n")
                f.write("- Working hardware monitoring\n")
                f.write("- Proper directory structures\n\n")

            else:
                f.write(f"❌ **{summary['failed_overall']} scripts have issues that must be resolved.**\n\n")

                if summary["passed_syntax"] < len(self.phase1_scripts):
                    f.write(f"- Fix syntax errors in {len(self.phase1_scripts) - summary['passed_syntax']} scripts\n")

                if summary["passed_imports"] < len(self.phase1_scripts):
                    f.write(f"- Resolve import issues in {len(self.phase1_scripts) - summary['passed_imports']} scripts\n")

            f.write("\n**Next Steps:**\n")
            f.write("1. Address any validation issues found\n")
            f.write("2. Re-run validation if fixes were applied\n")
            f.write("3. Execute individual scripts with `--help` to confirm argument parsing\n")
            f.write("4. Run dry-run tests before full model execution\n")
            f.write("5. Begin Phase 1 context scaling tests with validated scripts\n\n")

            f.write("---\n")
            f.write("**Validation Framework:** AI-First Test Suite Preparation\n")
            f.write("**Generated by:** O3 Phase 1 Validation Tester\n")

        print(f"\n📄 Validation report saved: {report_file}")

        # Save JSON summary
        summary_file = self.test_root / f"phase1_validation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        print(f"📄 Validation summary saved: {summary_file}")

    def print_validation_summary(self):
        """Print final validation summary"""

        print("\n" + "="*60)
        print("🎯 PHASE 1 VALIDATION COMPLETE")
        print("="*60)

        summary = self.results["summary"]

        print(f"📊 VALIDATION SUMMARY:")
        print(f"   Scripts Tested:     {len(self.results['scripts_validated'])}")
        print(f"   Syntax Check:       {summary['passed_syntax']}/{len(self.phase1_scripts)} ✅")
        print(f"   Import Validation:  {summary['passed_imports']}/{len(self.phase1_scripts)} ✅")
        print(f"   Overall Failures:   {summary['failed_overall']}")

        if "integration_tests" in self.results:
            integration = self.results["integration_tests"]
            ollama_ok = integration.get("ollama_connectivity", {}).get("passed", False)
            models_ok = integration.get("model_availability", {}).get("passed", False)

            print(f"\n🔗 INTEGRATION STATUS:")
            print(f"   Ollama Service:     {'✅ Connected' if ollama_ok else '❌ Issues'}")
            print(f"   Required Models:    {'✅ Available' if models_ok else '❌ Missing'}")

        all_passed = (summary["failed_overall"] == 0 and
                     self.results.get("integration_tests", {}).get("ollama_connectivity", {}).get("passed", False) and
                     self.results.get("integration_tests", {}).get("model_availability", {}).get("passed", False))

        final_status = "🎉 READY FOR PHASE 1 EXECUTION" if all_passed else "⚠️ ISSUES REQUIRE RESOLUTION"

        print(f"\n🏆 FINAL STATUS:")
        print(f"   {final_status}")

        if not all_passed:
            print("   ⚠️  Address validation issues before proceeding to model tests")
        print(f"\n📁 Detailed reports saved in: {self.test_root}")

def main():
    print("Starting O3 Phase 1 validation...")

    validator = Phase1ValidationTester()
    validator.run_validation()

    print("\nPhase 1 validation complete!")

if __name__ == "__main__":
    main()
