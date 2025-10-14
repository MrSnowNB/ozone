#!/usr/bin/env python3
"""
O3 Quick Start - Automated setup and initial test
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version}")

    # Check if Ollama is installed and running
    if not run_command("ollama --version", "Check Ollama installation"):
        print("❌ Ollama not found. Please install from https://ollama.ai")
        return False

    if not run_command("ollama list", "Check Ollama service"):
        print("❌ Ollama service not running. Please start with 'ollama serve'")
        return False

    return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Install Python dependencies")

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

    return run_command(f"python o3_optimizer.py {test_model} --concurrency 1", 
                      f"Sample optimization test")

def generate_sample_report():
    """Generate a sample report"""
    return run_command("python o3_report_generator.py --csv", "Generate sample report")

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
                if generate_sample_report():
                    print("\n📊 Sample report generated!")

                    # Show next steps
                    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                        Next Steps                            ║
╠══════════════════════════════════════════════════════════════╣
║ 1. Check results in 'o3_results/' directory                 ║
║ 2. View the generated report: O3_Report_*.md                ║
║ 3. Check optimized settings: o3_results/defaults/           ║
║ 4. Run full test suite:                                     ║
║    python o3_optimizer.py {' '.join(available_models[:3])}   ║
║                                                              ║
║ 🚀 VS Code users: Use Ctrl+Shift+P → "Tasks: Run Task"     ║
╚══════════════════════════════════════════════════════════════╝
                    """)
                else:
                    print("⚠️  Report generation failed, but test data is available")
            else:
                print("❌ Sample test failed")
        else:
            print("\nSetup complete! Run 'python o3_optimizer.py <model-name>' to start optimizing.")

    print("\n✅ O3 Quick Start completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
