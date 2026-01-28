#!/bin/bash
#SBATCH --job-name=ml_benchmark
#SBATCH --output=logs/ml_benchmark_%A_%a.out
#SBATCH --error=logs/ml_benchmark_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9

# Usage: sbatch run_ml_benchmark.sh <optimizer_type> <benchmark_name>
# Example: sbatch run_ml_benchmark.sh RS linear_regression
#
# Arguments:
#   $1: optimizer_type (RS or RE)
#   $2: benchmark_name (ML: linear_regression, logistic_regression, xor_mlp, autoencoder
#                       Synthetic: sphere, ellipsoid, rotated_quadratic, rosenbrock, rastrigin, ackley)
#
# The SLURM_ARRAY_TASK_ID is used as the seed
# To run with more seeds, adjust --array parameter (e.g., --array=0-29 for 30 seeds)

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: sbatch run_ml_benchmark.sh <optimizer_type> <benchmark_name>"
    echo "Example: sbatch run_ml_benchmark.sh RS linear_regression"
    exit 1
fi

OPTIMIZER_TYPE=$1
BENCHMARK_NAME=$2
SEED=$SLURM_ARRAY_TASK_ID

# Validate optimizer type
if [[ "$OPTIMIZER_TYPE" != "RS" && "$OPTIMIZER_TYPE" != "RE" ]]; then
    echo "Error: Invalid optimizer type '$OPTIMIZER_TYPE'. Must be 'RS' or 'RE'"
    exit 1
fi

# Validate benchmark name
VALID_BENCHMARKS=("linear_regression" "logistic_regression" "xor_mlp" "autoencoder" "sphere" "ellipsoid" "rotated_quadratic" "rosenbrock" "rastrigin" "ackley")
if [[ ! " ${VALID_BENCHMARKS[@]} " =~ " ${BENCHMARK_NAME} " ]]; then
    echo "Error: Invalid benchmark name '$BENCHMARK_NAME'"
    echo "Valid options: ${VALID_BENCHMARKS[@]}"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""
echo "Benchmark Configuration"
echo "=========================================="
echo "Optimizer: $OPTIMIZER_TYPE"
echo "Benchmark: $BENCHMARK_NAME"
echo "Seed: $SEED"
echo "=========================================="
echo ""

# Activate conda environment (adjust as needed)
# source ~/.bashrc
# conda activate your_env_name

# Or activate virtual environment
# source /path/to/venv/bin/activate

# Run the Python script
python nos_ml_benchmarks.py \
    --optimizer $OPTIMIZER_TYPE \
    --benchmark $BENCHMARK_NAME \
    --seed $SEED \
    --n_optimizer_samples 100 \
    --n_steps 500 \
    --n_samples 100 \
    --n_dims 5 \
    --weight_decay 0.01 \
    --results_dir ml_benchmark_results

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
