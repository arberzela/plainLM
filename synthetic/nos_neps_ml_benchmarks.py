"""
ML-based optimizer benchmarking script.
Tests optimizers on realistic ML tasks: linear regression, logistic regression,
noisy regression, XOR MLP, and tiny autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import argparse
from pathlib import Path
import neps
from neps import Trial
from neps.state import trial
from neps.space.neps_spaces import sampling, neps_space
from nos_space import NOSSpaceMaxLines
import torch.optim as optim
from functools import partial


# ----------------------------
# ML Benchmark Definitions
# ----------------------------

def get_ml_benchmark_function(name: str, seed: int = 0, n_samples: int = 100, n_dims: int = 5):
    """
    Factory function to get ML benchmark setup.
    
    Args:
        name: Name of the benchmark
        seed: Random seed for reproducibility
        n_samples: Number of data samples (for regression/classification tasks)
        n_dims: Number of input dimensions/features
    
    Returns:
        Tuple of (loss_fn, params_list, description)
        where loss_fn is a callable that returns a loss tensor,
        and params_list is a list of parameters to optimize.
    """
    torch.manual_seed(seed)
    
    if name == 'linear_regression':
        n, d = n_samples, n_dims
        X = torch.randn(n, d)
        true_w = torch.randn(d)
        y = X @ true_w + 0.1 * torch.randn(n)
        w = torch.randn(d, requires_grad=True)
        
        def loss_fn():
            return torch.mean((X @ w - y)**2)
        
        return loss_fn, [w], f"Linear Regression (Convex, n={n}, d={d})"
    
    elif name == 'logistic_regression':
        n, d = n_samples, n_dims
        X = torch.randn(n, d)
        true_w = torch.randn(d)
        logits = X @ true_w
        y = torch.bernoulli(torch.sigmoid(logits))
        w = torch.randn(d, requires_grad=True)
        
        def loss_fn():
            return F.binary_cross_entropy_with_logits(X @ w, y)
        
        return loss_fn, [w], f"Logistic Regression (Mildly Non-Convex, n={n}, d={d})"
    
    elif name == 'xor_mlp':
        # XOR problem is fixed at 2D input, but we can scale hidden layer with n_dims
        X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
        y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
        
        # Scale hidden layer size with n_dims parameter
        hidden_size = max(4, n_dims)
        model = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        def loss_fn():
            logits = model(X)
            return F.binary_cross_entropy_with_logits(logits, y)
        
        return loss_fn, list(model.parameters()), f"XOR MLP (Nonlinear, hidden={hidden_size})"
    
    elif name == 'autoencoder':
        n, d = n_samples, n_dims
        X = torch.randn(n, d)
        
        # Scale latent dimension with problem size
        dim_latent = max(2, d // 3)
        hidden_size = max(4, (d + dim_latent) // 2)
        
        model = nn.Sequential(
            nn.Linear(d, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_latent),
            nn.ReLU(),
            nn.Linear(dim_latent, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d)
        )
        
        def loss_fn():
            recon = model(X)
            return torch.mean((recon - X)**2)
        
        return loss_fn, list(model.parameters()), f"Tiny Autoencoder (Deep & Nonlinear, n={n}, d={d}, latent={dim_latent})"
    
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def evaluate_ml_pipeline(optimizer_cls, learning_rate, benchmark_name='linear_regression', 
                        n_steps=500, seed=0, n_samples=100, n_dims=5, weight_decay=0.0):
    """
    Evaluate an optimizer on an ML benchmark.
    
    Args:
        optimizer_cls: Optimizer class or creator
        learning_rate: Learning rate for the optimizer
        benchmark_name: Name of the ML benchmark
        n_steps: Number of optimization steps
        seed: Random seed for reproducibility
        n_samples: Number of data samples
        n_dims: Number of input dimensions
        weight_decay: Weight decay (L2 regularization) coefficient
    
    Returns:
        Tuple of (final_loss, loss_history)
    """
    loss_fn, params, _ = get_ml_benchmark_function(benchmark_name, seed=seed, 
                                                             n_samples=n_samples, n_dims=n_dims)
    
    # Create optimizer with weight_decay
    opt = optimizer_cls(params, lr=learning_rate, weight_decay=weight_decay)
    
    loss_history = []
    
    for _ in range(n_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))
    
    # Final loss
    with torch.no_grad():
        final_loss = float(loss_fn().item())
    
    return final_loss#, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimizer benchmarks on ML tasks.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default='linear_regression',
        help="ML benchmark name."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='RE',
        help="Optimizer to use."
    )
    parser.add_argument(
        "--n_optimizer_samples",
        type=int,
        default=100,
        help="Number of different optimizers to sample (random search iterations)."
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=500,
        help="Number of optimization steps per optimizer."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ml_benchmark_results_neps",
        help="Directory to write benchmark results."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for benchmark data generation."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of data samples (for regression/classification tasks)."
    )
    parser.add_argument(
        "--n_dims",
        type=int,
        default=5,
        help="Number of input dimensions/features."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization) coefficient for optimizers. Default: 0.01"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    space = NOSSpaceMaxLines(max_lines=10)
    if args.optimizer == 'RE':
        blackbox_optimizer = neps.algorithms.neps_regularized_evolution
    elif args.optimizer == 'RS':
        blackbox_optimizer = neps.algorithms.neps_random_search

    logging.basicConfig(level=logging.DEBUG)

    neps.run(
        evaluate_pipeline=partial(
            evaluate_ml_pipeline,
            benchmark_name=args.benchmark,
            n_steps=args.n_steps,
            seed=args.seed,
            n_samples=args.n_samples,
            n_dims=args.n_dims,
            weight_decay=args.weight_decay
        ),
        pipeline_space=space,
        root_directory=str(results_dir),
        evaluations_to_spend=args.n_optimizer_samples,
        optimizer=blackbox_optimizer,
    )
