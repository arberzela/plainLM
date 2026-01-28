# ML Benchmarks - Updated Structure

This directory contains scripts for benchmarking optimizers on **ML tasks and synthetic functions** with improved structure for running multiple seeds and generating statistical plots.

## Overview of Changes

The benchmark system has been refactored with the following improvements:

1. **Parameterized optimizer type**: Choose between RS (Random Search) or RE (Regularized Evolution)
2. **Single benchmark execution**: Run one benchmark at a time for better parallelization
3. **Individual result files**: Each optimizer-benchmark-seed combination saves to its own JSON file
4. **Incumbent tracking**: Track the best loss seen so far at each optimization step
5. **SLURM array jobs**: Parallel execution across seeds using SLURM array tasks
6. **Unified framework**: Support for both ML tasks (neural networks) and synthetic functions (mathematical optimizations)

## Supported Benchmarks

### ML Benchmarks
- `linear_regression`: Convex optimization problem
- `logistic_regression`: Binary classification with mild non-convexity
- `xor_mlp`: Nonlinear MLP on XOR problem
- `autoencoder`: Deep autoencoder with multiple layers

### Synthetic Benchmarks
- `sphere`: Simple quadratic function (convex)
- `ellipsoid`: Ill-conditioned quadratic (convex)
- `rotated_quadratic`: Rotated ellipsoid with condition number variation
- `rosenbrock`: Classic non-convex optimization benchmark
- `rastrigin`: Highly multimodal function with many local minima
- `ackley`: Multimodal with nearly flat outer region

## File Structure

```
synthetic/
├── nos_ml_benchmarks.py           # Main benchmark script
├── run_ml_benchmark.sh            # SLURM script for single job
├── submit_all_benchmarks.sh       # Helper to submit all combinations
├── load_and_plot_results.py       # Script to load results and generate plots
└── ml_benchmark_results/          # Results directory
    ├── linear_regression/
    │   ├── RS_0.json
    │   ├── RS_1.json
    │   ├── Adam_0.json
    │   ├── Adam_1.json
    │   └── ...
    ├── logistic_regression/
    │   └── ...
    └── ...
```

## Result File Format

Each JSON file (`$results_dir/$benchmark/$optimizer_$seed.json`) contains:

```json
{
  "optimizer": "RS",
  "benchmark": "linear_regression",
  "seed": 0,
  "learning_rate": 0.001,
  "n_steps": 500,
  "n_samples": 100,
  "n_dims": 5,
  "weight_decay": 0.01,
  "final_loss": 0.0123,
  "incumbent_history": [1.5, 1.2, 0.8, 0.5, 0.3, ...],
  "loss_history": [1.5, 1.3, 0.8, 0.9, 0.3, ...]
}
```

- **incumbent_history**: Best loss encountered up to each step (monotonically decreasing)
- **loss_history**: Actual loss at each step (may fluctuate)

## Usage

### Running Locally

Run a single benchmark with a specific optimizer and seed:

```bash
# ML benchmark
python nos_ml_benchmarks.py \
    --optimizer RS \
    --benchmark linear_regression \
    --seed 0 \
    --n_optimizer_samples 100 \
    --n_steps 500 \
    --results_dir ml_benchmark_results

# Synthetic benchmark
python nos_ml_benchmarks.py \
    --optimizer RS \
    --benchmark sphere \
    --seed 0 \
    --n_optimizer_samples 100 \
    --n_steps 500 \
    --n_dims 100 \
    --results_dir ml_benchmark_results
```

**Arguments:**
- `--optimizer`: Choose `RS` or `RE`
  - `RS`: Runs RS + Adam + AdamW + SGD (baselines use same sampled learning rate)
  - `RE`: Runs only RE optimizer
- `--benchmark`: One of:
  - ML: `linear_regression`, `logistic_regression`, `xor_mlp`, `autoencoder`
  - Synthetic: `sphere`, `ellipsoid`, `rotated_quadratic`, `rosenbrock`, `rastrigin`, `ackley`
- `--seed`: Random seed for reproducibility
- `--n_optimizer_samples`: Number of different optimizer configurations to sample
- `--n_steps`: Number of optimization steps per run
- `--n_samples`: Number of data samples for the ML task
- `--n_dims`: Number of input dimensions/features
- `--weight_decay`: L2 regularization coefficient

### Running on SLURM

#### Single Job with Array Tasks

Submit a single optimizer-benchmark combination with 10 seeds:

```bash
# ML benchmark
sbatch --array=0-9 run_ml_benchmark.sh RS linear_regression

# Synthetic benchmark
sbatch --array=0-9 run_ml_benchmark.sh RS sphere
```

The `SLURM_ARRAY_TASK_ID` is automatically used as the seed.

#### All Combinations

Submit all optimizer-benchmark combinations:

```bash
./submit_all_benchmarks.sh
```

This will submit SLURM array jobs for:
- RS + all baselines on each benchmark
- RE on each benchmark
- Each with N seeds (default: 10, configurable in the script)

**Monitor jobs:**
```bash
squeue -u $USER
```

**Cancel jobs:**
```bash
scancel -u $USER
```

### Loading and Plotting Results

After running experiments with multiple seeds, generate plots and statistics:

```bash
python load_and_plot_results.py \
    --benchmark linear_regression \
    --optimizers RS RE Adam AdamW SGD \
    --results_dir ml_benchmark_results \
    --output ml_plots/linear_regression_comparison.png
```

This will:
1. Load all results for each optimizer across all seeds
2. Compute mean incumbent curves with standard deviation
3. Generate a plot with error bars
4. Print a summary table with final loss statistics

**Example output:**
```
Summary for linear_regression
================================================================================
Optimizer       N Seeds    Mean Final Loss      Std Final Loss      
--------------------------------------------------------------------------------
RS              10         1.234567e-03         2.345678e-04        
RE              10         9.876543e-04         1.234567e-04        
Adam            10         1.111111e-03         3.333333e-04        
AdamW           10         1.222222e-03         2.222222e-04        
SGD             10         5.555555e-03         1.111111e-03        
================================================================================
```

## Key Behavior Differences

### RS Mode
When `--optimizer RS` is selected:
- Samples one NEPS optimizer configuration with random search
- Runs the NEPS RS optimizer
- Runs Adam, AdamW, and SGD baselines **with the same sampled learning rate**
- All four optimizers (RS + 3 baselines) are run for each sample

### RE Mode
When `--optimizer RE` is selected:
- Uses regularized evolution to sample optimizer configuration
- Runs **only** the NEPS RE optimizer
- Updates the trial population for the next iteration
- Does **not** run baseline optimizers

This design allows:
- Fair comparison of RS vs baselines (same learning rate)
- Efficient RE exploration (no wasted baseline runs)
- Easy parallelization across seeds via SLURM arrays

## Tips

1. **Start small**: Test with `--n_optimizer_samples 10` and `--n_steps 100` before full runs
2. **Multiple seeds**: Use at least 10 seeds for reliable statistics
3. **Learning rate**: The sampled learning rate is shared between RS and baselines for fair comparison
4. **Storage**: Each result file is small (~50KB), but many seeds × many samples can add up
5. **Plotting**: Generate plots after all seeds complete for best results
6. **Synthetic benchmarks**: Use higher `--n_dims` (e.g., 100-1000) for synthetic functions to make problems more challenging
7. **ML vs Synthetic**: ML benchmarks use `--n_samples` (dataset size), synthetic benchmarks ignore it and use `--n_dims` (problem dimension)

## Example Workflow

```bash
# 1. Submit all combinations with 30 seeds
# Edit submit_all_benchmarks.sh to set N_SEEDS=30
./submit_all_benchmarks.sh

# 2. Wait for jobs to complete
watch squeue -u $USER

# 3. Generate plots for each benchmark
for benchmark in linear_regression logistic_regression xor_mlp autoencoder sphere ellipsoid rosenbrock rastrigin ackley; do
    python load_and_plot_results.py \
        --benchmark $benchmark \
        --optimizers RS RE Adam AdamW SGD \
        --output ml_plots/${benchmark}_comparison.png
done

# 4. Review summary statistics
for benchmark in linear_regression logistic_regression xor_mlp autoencoder sphere ellipsoid rosenbrock rastrigin ackley; do
    python load_and_plot_results.py --benchmark $benchmark
done
```

## Troubleshooting

**No results found:**
- Check that the benchmark name matches exactly
- Verify the results directory path
- Ensure jobs completed successfully (check SLURM logs)

**Different curve lengths:**
- The plotting script automatically truncates to the minimum length
- Ensure `--n_steps` is consistent across runs

**Missing seeds:**
- Some SLURM array tasks may have failed
- Check error logs in `logs/ml_benchmark_*.err`
- Rerun specific seeds if needed
