#!/usr/bin/env python3
"""
O3 AI-First Optimizer - Final Comprehensive Validation Report
Complete Phase 1 & Phase 2 Validation with Production Readiness Assessment
"""

import json
import subprocess
import sys
from pathlib import Path
import datetime

def run_test_suite(test_file, test_name):
    """Run a test suite and capture results"""
    print(f"\n🔬 Running {test_name}...")

    try:
        # Run pytest and capture output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=300
        )

        # Parse results
        output_lines = result.stdout.split('\n')
        summary_line = None
        for line in output_lines:
            if 'passed' in line and 'failed' in line:
                summary_line = line
                break

        if summary_line:
            # Extract pass/fail counts
            parts = summary_line.split()
            passed = int(parts[0]) if parts[0].isdigit() else 0
            failed = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            total = passed + failed

            return {
                "test_name": test_name,
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0,
                "status": "PASS" if failed == 0 else "FAIL",
                "output": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
            }
        else:
            return {
                "test_name": test_name,
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0,
                "status": "ERROR",
                "error": result.stderr or "No summary found"
            }

    except subprocess.TimeoutExpired:
        return {
            "test_name": test_name,
            "status": "TIMEOUT",
            "error": "Test execution timed out"
        }
    except Exception as e:
        return {
            "test_name": test_name,
            "status": "ERROR",
            "error": str(e)
        }

def run_integration_demos():
    """Run integration demos and capture status"""
    demos = [
        {
            "name": "Phase 2 Hardware Intelligence Demo",
            "command": [sys.executable, "o3_optimizer_phase2_test.py", "--demo"]
        }
    ]

    results = []
    for demo in demos:
        print(f"\n🚀 Running {demo['name']}...")

        try:
            result = subprocess.run(
                demo['command'],
                capture_output=True, text=True, timeout=60
            )

            results.append({
                "demo_name": demo['name'],
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "output": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                "error": result.stderr
            })

        except subprocess.TimeoutExpired:
            results.append({
                "demo_name": demo['name'],
                "status": "TIMEOUT",
                "error": "Demo execution timed out"
            })
        except Exception as e:
            results.append({
                "demo_name": demo['name'],
                "status": "ERROR",
                "error": str(e)
            })

    return results

def analyze_architecture():
    """Analyze codebase architecture and completeness"""
    files_to_analyze = [
        "o3_optimizer.py",
        "hardware_monitor.py",
        "o3_ai_config.yaml",
        "test_o3_ai_optimizer.py",
        "o3_optimizer_phase2_test.py",
        "o3_optimizer_log.md"
    ]

    architecture_info = {
        "core_modules": [],
        "test_suites": [],
        "configuration_files": [],
        "documentation": []
    }

    for file in files_to_analyze:
        if Path(file).exists():
            file_info = {
                "name": file,
                "exists": True,
                "size_kb": Path(file).stat().st_size / 1024,
                "lines": sum(1 for _ in open(file, encoding='utf-8', errors='ignore')) if Path(file).suffix == '.py' else None
            }

            if file.endswith('.py'):
                if 'test' in file:
                    architecture_info["test_suites"].append(file_info)
                else:
                    architecture_info["core_modules"].append(file_info)
            elif file.endswith('.yaml'):
                architecture_info["configuration_files"].append(file_info)
            elif file.endswith('.md'):
                architecture_info["documentation"].append(file_info)
        else:
            architecture_info["missing_files"] = architecture_info.get("missing_files", [])
            architecture_info["missing_files"].append(file)

    return architecture_info

def generate_validation_report():
    """Generate comprehensive validation report"""

    print("O3 AI-First Optimizer - Final Validation Report")
    print("=" * 60)

    report = {
        "validation_timestamp": datetime.datetime.now().isoformat(),
        "o3_version": "Phase 2 Complete",
        "validation_type": "Production Readiness Assessment"
    }

    # Architecture Analysis
    print("\n📊 Analyzing Architecture...")
    arch_info = analyze_architecture()
    report["architecture"] = arch_info

    # Test Suite Execution
    print("\n🧪 Running Test Suites...")

    test_suites = [
        ("test_o3_ai_optimizer.py", "Phase 1 AI-First Core Tests"),
        ("o3_optimizer_phase2_test.py", "Phase 2 Hardware Intelligence Tests")
    ]

    test_results = []
    total_tests = 0
    total_passed = 0

    for test_file, test_name in test_suites:
        result = run_test_suite(test_file, test_name)
        test_results.append(result)

        if result["status"] == "PASS":
            total_tests += result["total_tests"]
            total_passed += result["passed"]
            print(f"✅ {result['test_name']}: {result['passed']}/{result['total_tests']} tests passed")
        else:
            print(f"❌ {result['test_name']}: {result['status']} - {result.get('error', 'Unknown error')}")

    # Integration Demos
    print("\n🚀 Running Integration Demos...")
    demo_results = run_integration_demos()

    for demo in demo_results:
        status_icon = "✅" if demo["status"] == "PASS" else "❌"
        print(f"{status_icon} {demo['demo_name']}: {demo['status']}")

    # Comprehensive Report
    report["test_results"] = test_results
    report["demo_results"] = demo_results

    # Overall Validation Metrics
    overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0

    validation_summary = {
        "total_tests_executed": total_tests,
        "total_tests_passed": total_passed,
        "overall_pass_rate": overall_pass_rate,
        "overall_status": "PASS" if overall_pass_rate == 1.0 else "FAIL",
        "phase1_tests": next((t["total_tests"] for t in test_results if "Phase 1" in t["test_name"]), 0),
        "phase1_passed": next((t["passed"] for t in test_results if "Phase 1" in t["test_name"]), 0),
        "phase2_tests": next((t["total_tests"] for t in test_results if "Phase 2" in t["test_name"]), 0),
        "phase2_passed": next((t["passed"] for t in test_results if "Phase 2" in t["test_name"]), 0),
        "all_demos_passed": all(d["status"] == "PASS" for d in demo_results)
    }

    report["validation_summary"] = validation_summary

    # Production Readiness Assessment
    production_readiness = {
        "code_stability": validation_summary["overall_status"] == "PASS",
        "test_coverage": bool(arch_info.get("test_suites")),
        "documentation": bool(arch_info.get("documentation")),
        "configuration": bool(arch_info.get("configuration_files")),
        "hardware_support": "AMD_GPU" in str(test_results) and "NVIDIA" in str(test_results),
        "safety_systems": any("threshold" in t["test_name"].lower() for t in test_results),
        "monitoring_systems": any("monitor" in t["test_name"].lower() for t in test_results),
        "integration_ready": validation_summary["all_demos_passed"]
    }

    production_readiness["overall_readiness"] = all(production_readiness.values())
    report["production_readiness"] = production_readiness

    # Save detailed report
    report_file = Path("validation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print Summary
    print("\n" + "=" * 60)
    print("🎯 FINAL VALIDATION SUMMARY")
    print("=" * 60)

    print("\n📈 Test Results:")
    print(f"   Phase 1 AI-First Core: {validation_summary['phase1_passed']}/{validation_summary['phase1_tests']} ✅")
    print(f"   Phase 2 Hardware Intelligence: {validation_summary['phase2_passed']}/{validation_summary['phase2_tests']} ✅")
    print(f"   Overall Pass Rate: {overall_pass_rate:.1%} ({total_passed}/{total_tests} tests)")

    print("\n🛠️  Architecture Analysis:")
    print(f"   Core Modules: {len(arch_info.get('core_modules', []))} files")
    print(f"   Test Suites: {len(arch_info.get('test_suites', []))} suites")
    print(f"   Configuration: {len(arch_info.get('configuration_files', []))} files")
    print(f"   Documentation: {len(arch_info.get('documentation', []))} files")

    print("\n🚀 Integration Status:")
    demos_status = "✅ PASS" if validation_summary["all_demos_passed"] else "❌ FAIL"
    print(f"   Integration Demos: {demos_status}")

    print("\n🏆 Production Readiness:")
    readiness_icon = "✅ PRODUCTION READY" if production_readiness["overall_readiness"] else "⚠️  REVIEW NEEDED"
    print(f"   Overall Status: {readiness_icon}")

    if production_readiness["overall_readiness"]:
        print("\n🎉🎉🎉 O3 AI-First Optimizer is PRODUCTION READY! 🎉🎉🎉")
        print("\nReady for:")
        print("🤖 Extreme context window optimization (32k/64k/128k+)")
        print("🖥️  Multi-GPU hardware exploitation (AMD/NVIDIA)")
        print("🛡️  Safe automated optimization with monitoring")
        print("⚡ High-performance agentic workflows")
    else:
        print("\n⚠️  Production readiness issues identified")
        for component, ready in production_readiness.items():
            if not ready and component != "overall_readiness":
                print(f"   - {component}: Needs attention")

    print(f"\n📄 Detailed report saved to: {report_file}")
    print("\n🔗 Ready to connect with Ollama for real-world optimization!")

    return report

if __name__ == "__main__":
    report = generate_validation_report()
