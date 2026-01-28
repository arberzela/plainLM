#!/bin/bash

# Quick test script to verify synthetic benchmarks work

echo "=========================================="
echo "Testing Synthetic Benchmarks Integration"
echo "=========================================="
echo ""

# Test 1: Sphere (simple convex)
echo "Test 1: Sphere benchmark"
python nos_ml_benchmarks.py \
    --optimizer RS \
    --benchmark sphere \
    --seed 0 \
    --n_dims 10 \
    --n_steps 50 \
    --n_optimizer_samples 2 \
    --results_dir test_synthetic_results

echo ""
echo "Test 2: Rastrigin benchmark"
python nos_ml_benchmarks.py \
    --optimizer RS \
    --benchmark rastrigin \
    --seed 0 \
    --n_dims 10 \
    --n_steps 50 \
    --n_optimizer_samples 2 \
    --results_dir test_synthetic_results

echo ""
echo "Test 3: Linear regression (ML benchmark)"
python nos_ml_benchmarks.py \
    --optimizer RS \
    --benchmark linear_regression \
    --seed 0 \
    --n_dims 5 \
    --n_samples 20 \
    --n_steps 50 \
    --n_optimizer_samples 2 \
    --results_dir test_synthetic_results

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  ls -lh test_synthetic_results/*/
echo ""
echo "View a result file:"
echo "  cat test_synthetic_results/sphere/RS_0.json | head -30"
echo ""
