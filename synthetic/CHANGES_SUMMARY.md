# Summary of Changes to nos_ml_benchmarks.py

## Overview
The script has been refactored to support parallelizable experiments across seeds with improved result tracking and storage.

## Key Changes

### 1. Parameterized Optimizer Type
- **Added**: `--optimizer` argument with choices `['RS', 'RE']`
- **Behavior**:
  - `RS`: Runs Random Search NEPS optimizer + Adam/AdamW/SGD baselines (all with same learning rate)
  - `RE`: Runs only Regularized Evolution NEPS optimizer
- **Rationale**: Allows fair comparison of RS with baselines while efficiently running RE without redundant baseline runs

### 2. Single Benchmark Execution
- **Changed**: `--benchmarks` (plural, list) → `--benchmark` (singular, string)
- **Type**: Now accepts a single benchmark choice from `['linear_regression', 'logistic_regression', 'xor_mlp', 'autoencoder']`
- **Rationale**: Enables parallel execution of different benchmarks as separate SLURM jobs

### 3. Individual Result Files
- **Old structure**: Single `all_results.json` file with nested data
- **New structure**: Individual files per optimizer-benchmark-seed
  - Path: `$results_dir/$benchmark/$optimizer_$seed.json`
  - Example: `ml_benchmark_results/linear_regression/RS_0.json`
- **Rationale**: 
  - Prevents file conflicts in parallel execution
  - Easier to identify missing seeds
  - Simpler to load and aggregate results

### 4. Incumbent Tracking
- **Added**: `incumbent_history` to each result file
- **Definition**: Best loss encountered up to each optimization step
- **Properties**: Monotonically non-increasing sequence
- **Purpose**: Enables plotting of convergence curves with proper error bars across seeds
- **Implementation**: Modified `evaluate_ml_pipeline()` to return incumbent history

### 5. Result File Format
Each JSON file now contains:
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
  "incumbent_history": [1.5, 1.2, 0.8, ...],  // Best so far
  "loss_history": [1.5, 1.3, 0.8, ...]         // Actual loss
}
```

### 6. SLURM Integration
Created three shell scripts:

#### `run_ml_benchmark.sh`
- SLURM array job script
- Takes optimizer and benchmark as positional arguments
- Uses `$SLURM_ARRAY_TASK_ID` as seed
- Example: `sbatch --array=0-9 run_ml_benchmark.sh RS linear_regression`

#### `submit_all_benchmarks.sh`
- Helper script to submit all combinations
- Configurable number of seeds
- Submits 2 optimizer types × 4 benchmarks × N seeds jobs

#### `plot_all_benchmarks.sh`
- Generate plots for all benchmarks after experiments complete
- Iterates over all benchmark names

### 7. Result Loading and Plotting
Created `load_and_plot_results.py` with functions:

#### `load_results_for_optimizer(results_dir, benchmark, optimizer)`
- Loads all JSON files for a specific optimizer-benchmark combination
- Aggregates across seeds
- Returns mean and std of incumbent curves

#### `plot_optimizer_comparison(results_dir, benchmark, optimizers, ...)`
- Plots mean incumbent curves with error bars
- Compares multiple optimizers on same benchmark
- Supports log scale

#### `generate_summary_table(results_dir, benchmark, optimizers)`
- Creates formatted table with final loss statistics
- Shows mean ± std across seeds for each optimizer

## Code Structure Changes

### Removed Functions
- `run_optimizer_on_benchmarks()`: Was designed for multiple benchmarks in one run

### Added Functions
- `run_single_optimizer()`: Simplified function for single optimizer-benchmark-seed run
- `load_results_for_optimizer()`: In load_and_plot_results.py
- `plot_optimizer_comparison()`: In load_and_plot_results.py
- `generate_summary_table()`: In load_and_plot_results.py

### Modified Functions
- `evaluate_ml_pipeline()`: Now returns `(final_loss, loss_history, incumbent_history)`

### Main Loop Changes
- Removed nested loop over benchmarks
- Simplified to single benchmark execution
- Split RS and RE logic into separate branches
- Removed global `all_results` dict (no longer needed)
- Each run saves immediately to individual file

## Migration Guide

### Old Usage
```bash
python nos_ml_benchmarks.py \
    --benchmarks linear_regression logistic_regression xor_mlp \
    --n_optimizer_samples 100 \
    --n_steps 500 \
    --seed 0
```
Result: One big `all_results.json` file

### New Usage
```bash
# Run RS with baselines on one benchmark
python nos_ml_benchmarks.py \
    --optimizer RS \
    --benchmark linear_regression \
    --seed 0 \
    --n_optimizer_samples 100 \
    --n_steps 500

# Or submit via SLURM with 10 seeds
sbatch --array=0-9 run_ml_benchmark.sh RS linear_regression
```
Result: 40 files (10 seeds × 4 optimizers) in `ml_benchmark_results/linear_regression/`

### Loading Results
```bash
# Generate plot and summary
python load_and_plot_results.py \
    --benchmark linear_regression \
    --optimizers RS RE Adam AdamW SGD
```

## Benefits

1. **Parallelization**: Each seed runs independently as SLURM array task
2. **Fault Tolerance**: Failed seeds don't affect others; easy to identify and rerun
3. **Storage Efficiency**: Can delete individual seed results after aggregation
4. **Debugging**: Easy to inspect results for specific seed-optimizer combinations
5. **Flexibility**: Can run different number of seeds for different benchmarks
6. **Fair Comparison**: RS and baselines use same learning rate per sample
7. **Statistical Rigor**: Mean ± std error bars from multiple seeds
8. **Scalability**: SLURM array jobs scale to thousands of seeds if needed

## Backward Compatibility

⚠️ **Breaking Changes**: 
- Old result files (`all_results.json`) are not compatible
- Need to rerun experiments to generate new format
- Plotting scripts from old version won't work with new results

## Testing Checklist

- [x] Script accepts `--optimizer` and `--benchmark` arguments
- [x] RS mode runs NEPS RS + 3 baselines with same LR
- [x] RE mode runs only NEPS RE
- [x] Results save to correct directory structure
- [x] Incumbent history tracks best loss correctly
- [x] SLURM script uses SLURM_ARRAY_TASK_ID as seed
- [x] Loading script aggregates across seeds correctly
- [x] Plotting generates mean curves with error bars
- [x] Summary table shows correct statistics
