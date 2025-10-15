"""O3 Optimization CLI Module"""

import argparse
import sys
from pathlib import Path


def optimize_command(args):
    """Run model optimization"""
    from ..core.optimizer import OllamaOptimizer

    print(f"Optimizing model: {args.model}")
    optimizer = OllamaOptimizer(args.output_dir)

    try:
        results = optimizer.test_model(args.model, concurrency_levels=[1, 2, 4])
        optimizer.save_results(args.model, results)
        print(f"Optimization complete. Results saved to {args.output_dir}")
    except Exception as e:
        print(f"Error during optimization: {e}", file=sys.stderr)
        sys.exit(1)


def add_optimize_parser(subparsers):
    """Add optimize subcommand parser"""
    parser = subparsers.add_parser("optimize", help="Optimize a single model")
    parser.add_argument("model", help="Model name to optimize")
    parser.add_argument(
        "--output-dir",
        default="results/current",
        help="Output directory for results"
    )
    parser.set_defaults(func=optimize_command)
