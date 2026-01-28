# Adding Synthetic Benchmarks to nos_ml_benchmarks.py

## Summary

The `nos_ml_benchmarks.py` script now supports both **ML benchmarks** (neural network training) and **synthetic benchmarks** (mathematical optimization functions) in a unified framework.

## What Was Added

### 1. Synthetic Benchmark Functions

Added `get_synthetic_benchmark_function()` that provides 6 classic optimization benchmarks:

- **sphere**: Simple quadratic function (convex baseline)
- **ellipsoid**: Ill-conditioned quadratic with varying scales
- **rotated_quadratic**: Rotated ellipsoid with random rotation matrix
- **rosenbrock**: Classic non-convex valley function
- **rastrigin**: Highly multimodal with many local minima
- **ackley**: Multimodal with nearly flat outer region

Each function returns:
- `fn(x)`: Objective function (returns scalar loss)
- `grad_fn(x)`: Analytical gradient function
- `start_pos`: Recommended starting position for optimization

### 2. Synthetic Evaluation Pipeline

Added `evaluate_synthetic_pipeline()` which:
- Creates torch Parameter from numpy starting position
- Iteratively computes loss and gradient using numpy functions
- Applies gradients via torch optimizer
- Tracks incumbent (best loss so far)
- Returns final loss, loss history, and incumbent history

### 3. Unified Interface

Modified `run_single_optimizer()` to automatically detect benchmark type:
```python
synthetic_benchmarks = ['sphere', 'ellipsoid', 'rotated_quadratic', 
                        'rosenbrock', 'rastrigin', 'ackley']
is_synthetic = benchmark_name in synthetic_benchmarks

if is_synthetic:
    # Use evaluate_synthetic_pipeline
else:
    # Use evaluate_ml_pipeline
```

### 4. Updated Argument Parser

Expanded `--benchmark` choices to include:
```python
choices=[
    # ML benchmarks
    'linear_regression', 'logistic_regression', 'xor_mlp', 'autoencoder',
    # Synthetic benchmarks
    'sphere', 'ellipsoid', 'rotated_quadratic', 'rosenbrock', 'rastrigin', 'ackley'
]
```

### 5. Updated SLURM Scripts

- `run_ml_benchmark.sh`: Added synthetic benchmarks to validation list
- `submit_all_benchmarks.sh`: Separates ML and synthetic benchmarks, submits all
- `plot_all_benchmarks.sh`: Generates plots for both types

### 6. Documentation Updates

Updated `README_UPDATED_STRUCTURE.md` with:
- List of supported synthetic benchmarks
- Example commands for running synthetic benchmarks
- Tips for choosing appropriate `--n_dims` for synthetic functions
- Updated example workflows

## Usage Examples

### Running Synthetic Benchmarks Locally

```bash
# Sphere function with 100 dimensions
python nos_ml_benchmarks.py \
    --optimizer RS \
    --benchmark sphere \
    --seed 0 \
    --n_dims 100 \
    --n_steps 500

# Rastrigin function with 50 dimensions
python nos_ml_benchmarks.py \
    --optimizer RE \
    --benchmark rastrigin \
    --seed 0 \
    --n_dims 50 \
    --n_steps 1000
```

### Running via SLURM

```bash
# Submit sphere benchmark with 20 seeds
sbatch --array=0-19 run_ml_benchmark.sh RS sphere

# Submit all synthetic benchmarks
# (Edit submit_all_benchmarks.sh first)
./submit_all_benchmarks.sh
```

## Key Differences: ML vs Synthetic

| Aspect | ML Benchmarks | Synthetic Benchmarks |
|--------|---------------|---------------------|
| **Loss computation** | PyTorch forward pass | Numpy function evaluation |
| **Gradient computation** | PyTorch autograd | Analytical gradient function |
| **Parameters** | Neural network weights | Single parameter vector |
| **Uses `--n_samples`** | Yes (dataset size) | No (ignored) |
| **Uses `--n_dims`** | Yes (feature dimension) | Yes (problem dimension) |
| **Typical dimensions** | 5-50 | 10-1000 |
| **Starting position** | Random initialization | Function-specific (from literature) |

## Result File Format

Both ML and synthetic benchmarks save to the same format with an additional field:

```json
{
  "optimizer": "RS",
  "benchmark": "sphere",
  "benchmark_type": "synthetic",  // NEW: "ml" or "synthetic"
  "seed": 0,
  "learning_rate": 0.001,
  "n_steps": 500,
  "n_dims": 100,
  "weight_decay": 0.01,
  "final_loss": 0.0123,
  "incumbent_history": [...],
  "loss_history": [...]
  // n_samples only present for ML benchmarks
}
```

## Advantages of Unified Framework

1. **Same infrastructure**: Both types use identical file structure, SLURM scripts, plotting tools
2. **Fair comparison**: Same optimizer sampling, same learning rates, same seed management
3. **Easy extension**: Add new benchmarks of either type by extending the appropriate function
4. **Reproducibility**: Synthetic functions use deterministic starting positions and seeds
5. **Scalability**: Synthetic benchmarks easily scale to high dimensions for stress testing

## Plotting

The `load_and_plot_results.py` script works identically for both types:

```bash
# Plot synthetic benchmark
python load_and_plot_results.py \
    --benchmark sphere \
    --optimizers RS RE Adam AdamW SGD

# Plot ML benchmark
python load_and_plot_results.py \
    --benchmark linear_regression \
    --optimizers RS RE Adam AdamW SGD
```

## Recommendations

1. **For quick testing**: Use `sphere` or `ellipsoid` with low dimensions (10-50)
2. **For challenging tests**: Use `rastrigin` or `ackley` with high dimensions (100-1000)
3. **For smooth optimization**: Use `sphere`, `ellipsoid`, `rosenbrock`
4. **For multimodal optimization**: Use `rastrigin`, `ackley`
5. **Typical settings**:
   - Dimensions: 100-500
   - Steps: 500-2000
   - Seeds: 10-30

## Testing

Verify the implementation works:

```bash
# Quick test on sphere (should converge quickly)
python nos_ml_benchmarks.py \
    --optimizer RS \
    --benchmark sphere \
    --seed 0 \
    --n_dims 10 \
    --n_steps 100 \
    --n_optimizer_samples 5

# Check results
ls ml_benchmark_results/sphere/
cat ml_benchmark_results/sphere/RS_0.json | head -20
```

## Code Structure

```
nos_ml_benchmarks.py
├── get_ml_benchmark_function()       # ML tasks (existing)
├── get_synthetic_benchmark_function()  # Synthetic functions (NEW)
├── evaluate_ml_pipeline()            # ML evaluation (existing)
├── evaluate_synthetic_pipeline()     # Synthetic evaluation (NEW)
├── run_single_optimizer()            # Unified runner (MODIFIED)
└── main execution loop               # Unchanged
```

## Backward Compatibility

✅ **Fully backward compatible**: All existing ML benchmark functionality remains unchanged. Synthetic benchmarks are additive.

## Migration from nos_synthetic.py

If you were using the standalone `nos_synthetic.py`:

**Old:**
```bash
python nos_synthetic.py \
    --benchmarks sphere ellipsoid \
    --dimensions 10 100 \
    --n_optimizer_samples 100
```

**New:**
```bash
# Run each benchmark-dimension combination separately
python nos_ml_benchmarks.py --optimizer RS --benchmark sphere --n_dims 10 --seed 0
python nos_ml_benchmarks.py --optimizer RS --benchmark sphere --n_dims 100 --seed 0
python nos_ml_benchmarks.py --optimizer RS --benchmark ellipsoid --n_dims 10 --seed 0
# etc...

# Or use SLURM for parallel execution
sbatch --array=0-9 run_ml_benchmark.sh RS sphere
sbatch --array=0-9 run_ml_benchmark.sh RS ellipsoid
```

The new structure is more parallelizable and provides better seed management.
