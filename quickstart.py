#!/usr/bin/env python3
"""
O3 Quick Start - Automated setup and initial test
"""

import subprocess
import sys
import os
import json
import glob
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False, ""

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version}")

    # Check if Ollama is installed and running
    success, _ = run_command("ollama --version", "Check Ollama installation")
    if not success:
        print("❌ Ollama not found. Please install from https://ollama.ai")
        return False

    success, _ = run_command("ollama list", "Check Ollama service")
    if not success:
        print("❌ Ollama service not running. Please start with 'ollama serve'")
        return False

    return True

def install_dependencies():
    """Install Python dependencies"""
    success, _ = run_command("pip install -r requirements.txt", "Install Python dependencies")
    return success

def detect_models():
    """Detect available models"""
    try:
        result = subprocess.run("ollama list", shell=True, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[1:]  # Skip header

        models = []
        target_models = [
            "qwen3-coder:30b", "orieg/gemma3-tools:27b-it-qat", 
            "liquid-rag:latest", "qwen2.5:3b-instruct", "gemma3:latest"
        ]

        for line in lines:
            if line.strip():
                model_name = line.split()[0]
                if model_name in target_models:
                    models.append(model_name)

        return models
    except subprocess.CalledProcessError:
        return []

def run_sample_test(models):
    """Run a sample optimization test"""
    if not models:
        print("❌ No supported models found")
        return False

    # Use smallest available model for quick test
    test_model = models[0]
    print(f"\n🧪 Running sample test with {test_model}...")

    success, _ = run_command(f"python o3_optimizer.py {test_model} --concurrency 1", 
                           f"Sample optimization test")
    return success

def generate_sample_report():
    """Generate a sample report"""
    success, _ = run_command("python o3_report_generator.py --csv", "Generate sample report")
    return success

def display_optimization_summary():
    """Display a human-readable summary of O3 optimization results"""
    print("\n" + "="*80)
    print("📊 O3 AI-First Optimization Summary")
    print("="*80)

    # Check for results directory
    results_dir = Path("o3_results")
    if not results_dir.exists():
        print("No optimization results found yet.")
        return

    # Display environment info
    env_files = list(results_dir.glob("env/env_*.json"))
    if env_files:
        try:
            with open(env_files[-1], 'r') as f:
                env_data = json.load(f)

            print(f"\n🖥️  System Configuration:")
            print(f"   • CPU: {env_data['cpu_info']['cores_physical']} cores ({env_data['cpu_info']['cores_logical']} logical)")
            print(f"   • RAM: {env_data['memory']['total_ram_gb']:.1f} GB total")
            print(f"   • GPU: {env_data['gpu_type'].title()}")
            print(f"   • Ollama: {env_data.get('ollama_version', 'Unknown').replace('ollama version is ', '')}")
        except Exception:
            pass

    # Display model results
    summary_files = list(results_dir.glob("summaries/*.json"))
    if not summary_files:
        print("\nNo optimization summaries found.")
        return

    try:
        # Show most recent summary
        with open(summary_files[-1], 'r') as f:
            summary = json.load(f)

        model = summary['model']
        print(f"\n🤖 Model: {model}")
        print(f"   📈 Tests: {summary['total_tests']} total, {summary['successful_tests']} successful")
        print(f"   ✅ AI Config: {'Used' if summary.get('ai_config_used') else 'Fallback'}")

        if 'presets' in summary and summary['presets']:
            print(f"\n🎯 Optimized Presets:")

            for preset_name, preset_data in summary['presets'].items():
                ctx = preset_data['num_ctx'] // 1024
                tps = preset_data['tokens_per_sec']
                ttft = preset_data['ttft_ms']
                stability = preset_data.get('stability_score', 0.0) * 100

                if preset_name == "max_context":
                    icon = "📏"
                    desc = "Maximum context for agent workflows"
                elif preset_name == "balanced":
                    icon = "⚖️"
                    desc = "Balanced performance"
                elif preset_name == "fast_response":
                    icon = "⚡"
                    desc = "Maximum speed"
                else:
                    icon = "🎯"
                    desc = preset_name

                print(f"\n   {icon} {preset_name.upper()}: {ctx}k context")
                print(f"      └─ Tokens/sec: {tps:.1f}, TTFT: {ttft:.0f}ms")
                print(f"      └─ Stability: {stability:.1f}%, Use case: {desc}")

    except Exception as e:
        print(f"Error reading summary: {e}")
        return

    # Show default YAML if available
    default_files = list(results_dir.glob("defaults/*.yaml"))
    if default_files:
        print(f"\n📁 Ready-to-Use Configurations:")
        for config_file in default_files:
            model_name = config_file.stem
            print(f"   • {model_name}: o3_results/defaults/{config_file.name}")

    # Provide actionable recommendations
    print(f"\n💡 Recommendations:")
    print(f"   1. For long conversations: Use 'max_context' preset")
    print(f"   2. For fast responses: Use 'fast_response' preset")
    print(f"   3. For balanced usage: Use 'balanced' preset")
    print(f"   4. Load configs: ollama run {model} --format o3_results/defaults/{model.replace(':', '_')}.yaml")
    print(f"   5. Run again for more models: python o3_optimizer.py <model-name>")

    print(f"\n" + "="*80)

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   O3 (Ozone) Quick Start                    ║
║              Ollama Open-Source Optimizer                   ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please resolve issues and try again.")
        return 1

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        return 1

    # Detect models
    print("\n🔍 Detecting available models...")
    available_models = detect_models()

    if available_models:
        print(f"✅ Found {len(available_models)} supported models:")
        for model in available_models:
            print(f"   - {model}")
    else:
        print("⚠️  No supported models found. You may need to pull models first:")
        print("   ollama pull qwen2.5:3b-instruct")
        print("   ollama pull gemma3:latest")

    # Offer to run sample test
    if available_models:
        response = input(f"\n❓ Run sample optimization test with {available_models[0]}? (y/N): ").lower()
        if response in ['y', 'yes']:
            if run_sample_test(available_models):
                print("\n🎉 Sample test completed successfully!")

                # Generate report
                generate_sample_report()  # Don't require this to succeed

            else:
                print("❌ Sample test failed")
        else:
            print("\nSetup complete! Run 'python o3_optimizer.py <model-name>' to start optimizing.")

    # Display optimization summary if available
    display_optimization_summary()

    print("\n✅ O3 Quick Start completed!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
