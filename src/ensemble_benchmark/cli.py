"""Command-line interface."""

import argparse

from .benchmarking import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark tree ensemble classifiers.")
    parser.add_argument("--quick", action="store_true", help="Use smaller tuning grids.")
    parser.add_argument("--output", default="reports", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    metrics, metadata = run_benchmark(args.output, args.quick, args.seed)
    print("\nExperiment complete. Metrics:")
    print(metrics.round(4).to_string(index=False))
    print(f"\nOutputs saved to: {args.output}")
    print(f"Quick mode: {metadata['quick_mode']}")


if __name__ == "__main__":
    main()
