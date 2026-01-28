"""
Helper script to load and aggregate results from the new JSON structure.
This script loads results saved as: results_dir/benchmark/optimizer_seed.json
and computes mean incumbent curves with error bars across seeds.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse


def load_results_for_optimizer(results_dir: str, benchmark: str, optimizer: str) -> Dict:
    """
    Load all results for a given optimizer-benchmark combination across all seeds.
    
    Args:
        results_dir: Base results directory
        benchmark: Benchmark name
        optimizer: Optimizer name (e.g., 'RS', 'RE', 'Adam', 'AdamW', 'SGD')
    
    Returns:
        Dict with keys:
            - 'incumbent_curves': List of incumbent histories (one per seed)
            - 'seeds': List of seed values
            - 'mean_incumbent': Mean incumbent curve across seeds
            - 'std_incumbent': Std deviation of incumbent curve across seeds
            - 'n_seeds': Number of seeds found
    """
    results_path = Path(results_dir) / benchmark
    
    if not results_path.exists():
        print(f"Warning: Results directory not found: {results_path}")
        return None
    
    # Find all JSON files for this optimizer
    pattern = f"{optimizer}_*.json"
    result_files = list(results_path.glob(pattern))
    
    if not result_files:
        print(f"Warning: No results found for {optimizer} on {benchmark}")
        return None
    
    incumbent_curves = []
    seeds = []
    
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            if 'incumbent_history' in data and data['incumbent_history']:
                incumbent_curves.append(data['incumbent_history'])
                seeds.append(data['seed'])
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
    
    if not incumbent_curves:
        return None
    
    # Convert to numpy array for easier computation
    # Handle different lengths by truncating to minimum length
    min_length = min(len(curve) for curve in incumbent_curves)
    incumbent_array = np.array([curve[:min_length] for curve in incumbent_curves])
    
    mean_incumbent = np.mean(incumbent_array, axis=0)
    std_incumbent = np.std(incumbent_array, axis=0)
    
    return {
        'incumbent_curves': incumbent_curves,
        'seeds': seeds,
        'mean_incumbent': mean_incumbent,
        'std_incumbent': std_incumbent,
        'n_seeds': len(seeds),
        'min_length': min_length
    }


def plot_optimizer_comparison(results_dir: str, benchmark: str, 
                             optimizers: List[str], 
                             output_file: str = None,
                             use_log_scale: bool = True):
    """
    Plot comparison of multiple optimizers on a single benchmark.
    
    Args:
        results_dir: Base results directory
        benchmark: Benchmark name
        optimizers: List of optimizer names to compare
        output_file: Path to save the plot (if None, displays interactively)
        use_log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))
    
    for idx, optimizer in enumerate(optimizers):
        results = load_results_for_optimizer(results_dir, benchmark, optimizer)
        
        if results is None:
            continue
        
        mean_curve = results['mean_incumbent']
        std_curve = results['std_incumbent']
        n_seeds = results['n_seeds']
        
        steps = np.arange(len(mean_curve))
        
        # Plot mean with shaded error bars
        plt.plot(steps, mean_curve, label=f"{optimizer} (n={n_seeds})", 
                color=colors[idx], linewidth=2)
        plt.fill_between(steps, 
                        mean_curve - std_curve, 
                        mean_curve + std_curve,
                        alpha=0.2, color=colors[idx])
    
    plt.xlabel('Optimization Steps', fontsize=12)
    plt.ylabel('Best Loss (Incumbent)', fontsize=12)
    plt.title(f'Optimizer Comparison on {benchmark}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if use_log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def generate_summary_table(results_dir: str, benchmark: str, 
                          optimizers: List[str]) -> str:
    """
    Generate a summary table with final loss statistics for each optimizer.
    
    Args:
        results_dir: Base results directory
        benchmark: Benchmark name
        optimizers: List of optimizer names
    
    Returns:
        Formatted string table
    """
    lines = []
    lines.append(f"\nSummary for {benchmark}")
    lines.append("=" * 80)
    lines.append(f"{'Optimizer':<15} {'N Seeds':<10} {'Mean Final Loss':<20} {'Std Final Loss':<20}")
    lines.append("-" * 80)
    
    for optimizer in optimizers:
        results = load_results_for_optimizer(results_dir, benchmark, optimizer)
        
        if results is None:
            lines.append(f"{optimizer:<15} {'N/A':<10} {'N/A':<20} {'N/A':<20}")
            continue
        
        final_losses = [curve[-1] for curve in results['incumbent_curves']]
        mean_final = np.mean(final_losses)
        std_final = np.std(final_losses)
        n_seeds = results['n_seeds']
        
        lines.append(f"{optimizer:<15} {n_seeds:<10} {mean_final:<20.6e} {std_final:<20.6e}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot ML benchmark results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ml_benchmark_results",
        help="Base results directory"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark name"
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=['RS', 'RE', 'Adam', 'AdamW', 'SGD'],
        help="List of optimizers to compare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for plot (if not specified, displays interactively)"
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        default=True,
        help="Use log scale for y-axis"
    )
    
    args = parser.parse_args()
    
    # Generate summary table
    summary = generate_summary_table(args.results_dir, args.benchmark, args.optimizers)
    print(summary)
    
    # Generate plot
    if args.output is None:
        output_file = f"ml_plots/{args.benchmark}_comparison.png"
    else:
        output_file = args.output
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    plot_optimizer_comparison(
        args.results_dir,
        args.benchmark,
        args.optimizers,
        output_file=output_file,
        use_log_scale=args.log_scale
    )
