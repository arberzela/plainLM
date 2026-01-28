#!/bin/bash

# Helper script to submit all benchmark combinations
# This will launch SLURM array jobs for each optimizer-benchmark combination

# Number of seeds (SLURM array size)
N_SEEDS=10  # Adjust this to run with more seeds (e.g., 30 for --array=0-29)

# Optimizer types
OPTIMIZERS=("RS" "RE")

# ML Benchmark names
ML_BENCHMARKS=("linear_regression" "logistic_regression" "xor_mlp" "autoencoder")

# Synthetic Benchmark names
SYNTHETIC_BENCHMARKS=("sphere" "ellipsoid" "rotated_quadratic" "rosenbrock" "rastrigin" "ackley")

# Combine all benchmarks
BENCHMARKS=("${ML_BENCHMARKS[@]}" "${SYNTHETIC_BENCHMARKS[@]}")

echo "=========================================="
echo "Submitting all benchmark combinations"
echo "=========================================="
echo "Number of seeds per combination: $N_SEEDS"
echo "ML Benchmarks: ${ML_BENCHMARKS[@]}"
echo "Synthetic Benchmarks: ${SYNTHETIC_BENCHMARKS[@]}"
echo ""

# Create logs directory
mkdir -p logs

# Submit jobs for each combination
for optimizer in "${OPTIMIZERS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        echo "Submitting: $optimizer on $benchmark"
        sbatch --array=0-$((N_SEEDS-1)) run_ml_benchmark.sh $optimizer $benchmark
    done
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Total combinations: $((${#OPTIMIZERS[@]} * ${#BENCHMARKS[@]}))"
echo "Total runs: $((${#OPTIMIZERS[@]} * ${#BENCHMARKS[@]} * N_SEEDS))"
echo "=========================================="
echo ""
echo "To check job status: squeue -u \$USER"
echo "To cancel all jobs: scancel -u \$USER"
