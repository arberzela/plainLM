#!/bin/bash

# Script to generate plots for all benchmarks after experiments complete

RESULTS_DIR=${1:-"ml_benchmark_results"}
OUTPUT_DIR=${2:-"ml_plots"}

# ML benchmarks
ML_BENCHMARKS=("linear_regression" "logistic_regression" "xor_mlp" "autoencoder")

# Synthetic benchmarks
SYNTHETIC_BENCHMARKS=("sphere" "ellipsoid" "rotated_quadratic" "rosenbrock" "rastrigin" "ackley")

# Combine all benchmarks
BENCHMARKS=("${ML_BENCHMARKS[@]}" "${SYNTHETIC_BENCHMARKS[@]}")

OPTIMIZERS=("RS" "RE" "Adam" "AdamW" "SGD")

echo "=========================================="
echo "Generating plots for all benchmarks"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "ML Benchmarks: ${ML_BENCHMARKS[@]}"
echo "Synthetic Benchmarks: ${SYNTHETIC_BENCHMARKS[@]}"
echo ""

mkdir -p $OUTPUT_DIR

for benchmark in "${BENCHMARKS[@]}"; do
    echo "Processing $benchmark..."
    
    # Generate plot and summary
    python load_and_plot_results.py \
        --benchmark $benchmark \
        --optimizers ${OPTIMIZERS[@]} \
        --results_dir $RESULTS_DIR \
        --output $OUTPUT_DIR/${benchmark}_comparison.png
    
    echo ""
done

echo "=========================================="
echo "All plots generated in $OUTPUT_DIR"
echo "=========================================="
