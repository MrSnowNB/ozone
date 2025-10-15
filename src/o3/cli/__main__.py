"""O3 Main CLI Entry Point"""

import argparse
import sys

from .optimize import add_optimize_parser


def create_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        prog="o3",
        description="O3 (Ozone) - AI-First Ollama Hardware Optimizer"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="O3 1.0.0"
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands"
    )

    # Add subcommand parsers
    add_optimize_parser(subparsers)

    # Add test command placeholder
    def test_command(args):
        print("Test command - implementation pending")
        # from .test import add_test_parser  # Future

    test_parser = subparsers.add_parser("test", help="Run test suites")
    test_parser.set_defaults(func=test_command)

    # Add report command placeholder
    def report_command(args):
        print("Report command - implementation pending")
        # from .report import add_report_parser  # Future

    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.set_defaults(func=report_command)

    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
