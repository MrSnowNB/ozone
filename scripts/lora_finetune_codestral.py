#!/usr/bin/env python3
"""
Lightweight LoRA Fine-Tuning for Codestral-22B Agentic Coding

This script implements AI-First fine-tuning approach:
1. Minimal dataset curation using in-context examples
2. LoRA adaptation for 256K context agentic tasks
3. Optimized for RTX A6000 multi-GPU setup

Dependencies: pip install torch transformers accelerate peft datasets

Usage:
python scripts/lora_finetune_codestral.py --model_id mistralai/Codestral-22B-v0.1 \
                                         --output_dir results/codestral_o3_adapter/
"""

import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def create_agentic_dataset():
    """Create AI-generated training dataset for agentic coding tasks"""

    agentic_examples = [
        {
            "instruction": "Write a Python function to analyze code complexity using AST parsing",
            "context": "You are an AI coding assistant. Provide a tool-using implementation.",
            "response": """```python
import ast
from typing import Dict

class ComplexityAnalyzer:
    \"\"\"Analyzes code complexity metrics using AST parsing\"\"\"

    def analyze_function_complexity(self, source_code: str) -> Dict[str, int]:
        \"\"\"Parse AST and calculate complexity metrics\"\"\"
        tree = ast.parse(source_code)

        complexity = {
            'cyclomatic': self._calculate_cyclomatic_complexity(tree),
            'functions': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
            'classes': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        }

        return complexity

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        \"\"\"Basic cyclomatic complexity calculation\"\"\"
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp) and len(node.values) > 1:
                complexity += len(node.values) - 1
        return complexity
```"""
        },
        {
            "instruction": "Implement a file watching tool that triggers code analysis on changes",
            "context": "Use watchdog library for file monitoring and integrate with your analyzer",
            "response": """```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeAnalyzerHandler(FileSystemEventHandler):
    \"\"\"File system event handler for automatic code analysis\"\"\"

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def on_modified(self, event):
        if event.src_path.endswith(('.py', '.js', '.ts')):
            print(f"Analyzing {event.src_path}...")
            with open(event.src_path, 'r') as f:
                complexity = self.analyzer.analyze_function_complexity(f.read())
                self._log_analysis(event.src_path, complexity)

    def _log_analysis(self, file_path: str, complexity: Dict[str, int]):
        \"\"\"Log analysis results\"\"\"
        print(f"Analysis complete for {file_path}:")
        print(f"- Cyclomatic complexity: {complexity.get('cyclomatic', 0)}")
        print(f"- Functions found: {complexity.get('functions', 0)}")
        print(f"- Classes found: {complexity.get('classes', 0)}")

def start_watching(project_path: str, analyzer):
    \"\"\"Start file monitoring for a project\"\"\"
    observer = Observer()
    handler = CodeAnalyzerHandler(analyzer)
    observer.schedule(handler, project_path, recursive=True)

    print(f"Starting file watcher on {project_path}")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\\nStopping file watcher...")
    observer.join()
```"""
        }
    ]

    # Format for training
    formatted_data = []
    for example in agentic_examples:
        system_prompt = example["context"]
        user_input = example["instruction"]
        assistant_response = example["response"]

        # Codestral chat format
        chat_text = f"<s>[INST] {system_prompt}\n{user_input} [/INST] {assistant_response} </s>"

        formatted_data.append({"text": chat_text})

    return Dataset.from_list(formatted_data)

def setup_quantization():
    """Configure 4-bit quantization for efficient training"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

def setup_lora_config():
    """Configure LoRA parameters for agentic task adaptation"""
    return LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def main():
    # Configuration
    MODEL_ID = "mistralai/Codestral-22B-v0.1"  # HuggingFace version
    OUTPUT_DIR = "results/codestral_o3_adapter/"
    NUM_EPOCHS = 2  # Lightweight training

    print("Starting AI-First LoRA Fine-Tuning for Codestral Agentic Coding...")

    # Setup device
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"

    # Create dataset
    print("Creating agentic coding dataset...")
    train_dataset = create_agentic_dataset()

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=2048
        )

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    # Setup model with quantization
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=setup_quantization(),
        device_map=device_map,
        trust_remote_code=True
    )

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, setup_lora_config())
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=25,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        remove_unused_columns=False,
        run_name="codestral_o3_lora"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Train
    print("Starting LoRA training...")
    trainer.train()

    # Save adapter
    print("Saving LoRA adapter...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete! Adapter saved to {OUTPUT_DIR}")

    # Export to Ollama format if needed
    print("To use in Ollama: ollama create codestral-o3 -f Modelfile")

if __name__ == "__main__":
    # Check requirements
    required_packages = ["torch", "transformers", "accelerate", "peft", "datasets"]
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        exit(1)

    main()
