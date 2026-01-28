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


# ----------------------------
# Synthetic Benchmark Definitions
# ----------------------------

def get_synthetic_benchmark_function(name: str, n_dims: int, seed: int = 42):
    """
    Factory function to get synthetic objective function and its gradient.
    
    Args:
        name: Name of the benchmark function
        n_dims: Number of dimensions
        seed: Random seed for reproducibility (used for some functions)
    
    Returns:
        Tuple of (fn, grad_fn, start_pos)
        where fn computes the loss, grad_fn computes the gradient,
        and start_pos is the recommended starting position.
    """
    if name == 'sphere':
        def fn(x):
            return np.sum(x**2)
        def grad(x):
            return 2 * x
        start_pos = np.full(n_dims, 5.0)

    elif name == 'ellipsoid':
        c = 10**np.linspace(0, 6, n_dims)
        def fn(x):
            return np.sum(c * (x**2))
        def grad(x):
            return 2 * c * x
        start_pos = np.full(n_dims, 5.0)

    elif name == 'rotated_quadratic':
        rng = np.random.RandomState(seed)
        M = rng.rand(n_dims, n_dims)
        Q, _ = np.linalg.qr(M)
        D = np.diag(10**np.linspace(0, 3, n_dims))
        A = Q.T @ D @ Q
        b = rng.rand(n_dims)

        def fn(x):
            return 0.5 * x.T @ A @ x - b.T @ x
        def grad(x):
            return A @ x - b
        start_pos = np.zeros(n_dims)

    elif name == 'rosenbrock':
        def fn(x):
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        def grad(x):
            g = np.zeros_like(x)
            g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
            g[1:] += 200 * (x[1:] - x[:-1]**2)
            return g
        start_pos = np.zeros(n_dims)

    elif name == 'rastrigin':
        def fn(x):
            return 10 * n_dims + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        def grad(x):
            return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
        start_pos = np.random.RandomState(seed).uniform(-5.12, 5.12, n_dims)

    elif name == 'ackley':
        def fn(x):
            term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
            term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
            return term1 + term2 + 20 + np.e
        def grad(x):
            s = np.sqrt(np.mean(x**2))
            if s == 0:
                return np.zeros_like(x)
            g1 = -20 * np.exp(-0.2*s) * (-0.2 * (0.5/s) * (2*x/n_dims))
            g2 = -np.exp(np.mean(np.cos(2*np.pi*x))) * (-2*np.pi/n_dims) * np.sin(2*np.pi*x)
            return g1 + g2
        start_pos = np.random.RandomState(seed).uniform(-32.7, 32.7, n_dims)

    else:
        raise ValueError(f"Unknown synthetic benchmark: {name}")

    return fn, grad, start_pos


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
        Tuple of (final_loss, loss_history, incumbent_history)
        incumbent_history tracks the best loss seen so far at each step
    """
    loss_fn, params, description = get_ml_benchmark_function(benchmark_name, seed=seed, 
                                                             n_samples=n_samples, n_dims=n_dims)
    
    # Create optimizer with weight_decay
    opt = optimizer_cls(params, lr=learning_rate, weight_decay=weight_decay)
    
    loss_history = []
    incumbent_history = []
    best_loss = float('inf')
    
    for step in range(n_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        
        current_loss = float(loss.item())
        loss_history.append(current_loss)
        
        # Track incumbent (best loss so far)
        if current_loss < best_loss:
            best_loss = current_loss
        incumbent_history.append(best_loss)
    
    # Final loss
    with torch.no_grad():
        final_loss = float(loss_fn().item())
    
    return final_loss, loss_history, incumbent_history

def evaluate_synthetic_pipeline(optimizer_cls, learning_rate, benchmark_name='sphere',
                               n_dims=10, n_steps=500, seed=42, weight_decay=0.0):
    """
    Evaluate an optimizer on a synthetic benchmark function.
    
    Args:
        optimizer_cls: Optimizer class or creator
        learning_rate: Learning rate for the optimizer
        benchmark_name: Name of the synthetic benchmark
        n_dims: Number of dimensions
        n_steps: Number of optimization steps
        seed: Random seed for reproducibility
        weight_decay: Weight decay (L2 regularization) coefficient
    
    Returns:
        Tuple of (final_loss, loss_history, incumbent_history)
        incumbent_history tracks the best loss seen so far at each step
    """
    # Get benchmark function and starting position
    fn, grad_fn, start_pos = get_synthetic_benchmark_function(benchmark_name, n_dims, seed)
    
    # Create parameter as torch tensor
    x = torch.nn.Parameter(torch.from_numpy(start_pos).float(), requires_grad=True)
    opt = optimizer_cls([x], lr=learning_rate, weight_decay=weight_decay)
    
    loss_history = []
    incumbent_history = []
    best_loss = float('inf')

    for step in range(n_steps):
        opt.zero_grad()

        # Compute loss and gradient using numpy functions
        x_np = x.detach().numpy()
        loss_value = fn(x_np)
        loss_history.append(float(loss_value))
        
        # Track incumbent (best loss so far)
        if loss_value < best_loss:
            best_loss = loss_value
        incumbent_history.append(best_loss)

        # Compute gradient and assign to parameter
        grad_np = grad_fn(x_np)
        x.grad = torch.from_numpy(grad_np).float()
        opt.step()

    # Final loss
    final_x_np = x.detach().numpy()
    final_loss = fn(final_x_np)

    return final_loss, loss_history, incumbent_history

def run_single_optimizer(opt_creator, learning_rate, optimizer_name, benchmark_name, 
                        n_steps, seed, n_samples, n_dims, weight_decay, output_dir):
    """
    Run a single optimizer on a benchmark and save results.
    
    Args:
        opt_creator: Function to create optimizer
        learning_rate: Learning rate
        optimizer_name: Name identifier for the optimizer
        benchmark_name: Name of the benchmark
        n_steps: Number of optimization steps
        seed: Random seed
        n_samples: Number of data samples (for ML tasks, ignored for synthetic)
        n_dims: Number of input dimensions
        weight_decay: Weight decay coefficient
        output_dir: Directory to save results
    
    Returns:
        Dict with final_loss and incumbent_history
    """
    print(f"  Running {optimizer_name} (lr={learning_rate:.6e})...")
    
    # List of synthetic benchmarks
    synthetic_benchmarks = ['sphere', 'ellipsoid', 'rotated_quadratic', 'rosenbrock', 'rastrigin', 'ackley']
    is_synthetic = benchmark_name in synthetic_benchmarks
    
    try:
        if is_synthetic:
            # Use synthetic evaluation
            final_loss, loss_history, incumbent_history = evaluate_synthetic_pipeline(
                optimizer_cls=opt_creator,
                learning_rate=learning_rate,
                benchmark_name=benchmark_name,
                n_dims=n_dims,
                n_steps=n_steps,
                seed=seed,
                weight_decay=weight_decay
            )
        else:
            # Use ML evaluation
            final_loss, loss_history, incumbent_history = evaluate_ml_pipeline(
                optimizer_cls=opt_creator,
                learning_rate=learning_rate,
                benchmark_name=benchmark_name,
                n_steps=n_steps,
                seed=seed,
                n_samples=n_samples,
                n_dims=n_dims,
                weight_decay=weight_decay
            )
        
        # Save results
        result_data = {
            'optimizer': optimizer_name,
            'benchmark': benchmark_name,
            'benchmark_type': 'synthetic' if is_synthetic else 'ml',
            'seed': seed,
            'learning_rate': float(learning_rate),
            'n_steps': n_steps,
            'n_dims': n_dims,
            'weight_decay': weight_decay,
            'final_loss': float(final_loss),
            'incumbent_history': [float(x) for x in incumbent_history],
            'loss_history': [float(x) for x in loss_history]
        }
        
        if not is_synthetic:
            result_data['n_samples'] = n_samples
        
        output_file = output_dir / f"{optimizer_name}_{seed}.json"
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"    Final loss: {final_loss:.6e} -> Saved to {output_file.name}")
        
        return result_data
    
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        result_data = {
            'optimizer': optimizer_name,
            'benchmark': benchmark_name,
            'benchmark_type': 'synthetic' if is_synthetic else 'ml',
            'seed': seed,
            'error': str(e),
            'final_loss': float('inf'),
            'incumbent_history': []
        }
        return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimizer benchmarks on ML tasks and synthetic functions.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default='linear_regression',
        choices=[
            # ML benchmarks
            'linear_regression', 'logistic_regression', 'xor_mlp', 'autoencoder',
            # Synthetic benchmarks
            'sphere', 'ellipsoid', 'rotated_quadratic', 'rosenbrock', 'rastrigin', 'ackley'
        ],
        help="Benchmark name (ML task or synthetic function)."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='RS',
        choices=['RS', 'RE'],
        help="Optimizer type: RS (random search) or RE (regularized evolution)."
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
        default="ml_benchmark_results",
        help="Directory to write benchmark results."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for benchmark data generation and optimizer sampling."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of data samples (for ML benchmarks only, ignored for synthetic functions)."
    )
    parser.add_argument(
        "--n_dims",
        type=int,
        default=5,
        help="Number of input dimensions/features. For ML: feature dimension. For synthetic: problem dimension."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization) coefficient for optimizers. Default: 0.01"
    )

    args = parser.parse_args()

    benchmark = args.benchmark
    optimizer_type = args.optimizer
    n_optimizer_samples = args.n_optimizer_samples
    n_steps = args.n_steps
    seed = args.seed
    n_samples = args.n_samples
    n_dims = args.n_dims
    weight_decay = args.weight_decay

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    results_dir = Path(args.results_dir)
    benchmark_dir = results_dir / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running benchmark: {benchmark}")
    print(f"Optimizer type: {optimizer_type}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    space = NOSSpaceMaxLines(max_lines=10)
    
    # Initialize samplers and trials for RE
    random_sampler = sampling.RandomSampler({})
    prior_sampler = sampling.PriorOrFallbackSampler(random_sampler)
    trials = {} if optimizer_type == 'RE' else None

    for optimizer_idx in range(n_optimizer_samples):
        print(f"\n{'='*60}")
        print(f"Optimizer Sample {optimizer_idx + 1}/{n_optimizer_samples}")
        print(f"{'='*60}")

        if optimizer_type == 'RS':
            # Random Search: Sample optimizer and learning rate
            resolved_pipeline, resolution_context = neps_space.resolve(space, domain_sampler=prior_sampler)
            optimizer_creator_object = resolved_pipeline.optimizer_cls
            learning_rate = resolved_pipeline.learning_rate
            optimizer_creator = neps_space.convert_operation_to_callable(optimizer_creator_object)
            
            print(f"Learning rate: {learning_rate}")
            
            # Run NEPS RS optimizer
            def neps_opt_wrapper(params, lr, weight_decay=0.0):
                return optimizer_creator(params, lr=lr, variables=(1, 1))
            
            neps_creator = partial(neps_opt_wrapper)
            run_single_optimizer(
                opt_creator=neps_creator,
                learning_rate=learning_rate,
                optimizer_name='RS',
                benchmark_name=benchmark,
                n_steps=n_steps,
                seed=seed,
                n_samples=n_samples,
                n_dims=n_dims,
                weight_decay=weight_decay,
                output_dir=benchmark_dir
            )
            
            # Run baseline optimizers with the same learning rate
            baseline_optimizers = {
                'Adam': lambda params, lr, weight_decay: optim.Adam(params, lr=lr, weight_decay=weight_decay),
                'AdamW': lambda params, lr, weight_decay: optim.AdamW(params, lr=lr, weight_decay=weight_decay),
                'SGD': lambda params, lr, weight_decay: optim.SGD(params, lr=lr, weight_decay=weight_decay)
            }
            
            for base_name, base_creator in baseline_optimizers.items():
                baseline_creator = lambda params, lr, weight_decay=weight_decay: base_creator(params, lr, weight_decay)
                run_single_optimizer(
                    opt_creator=baseline_creator,
                    learning_rate=learning_rate,
                    optimizer_name=base_name,
                    benchmark_name=benchmark,
                    n_steps=n_steps,
                    seed=seed,
                    n_samples=n_samples,
                    n_dims=n_dims,
                    weight_decay=weight_decay,
                    output_dir=benchmark_dir
                )
        
        elif optimizer_type == 'RE':
            # Regularized Evolution: Sample using NEPS RE algorithm
            config = neps.algorithms.neps_regularized_evolution(NOSSpaceMaxLines(), population_size=20, tournament_size=5)(trials, None)
            assert not isinstance(config, list)
            samplings = neps_space.NepsCompatConverter().from_neps_config(config.config).predefined_samplings

            resolved_pipeline, resolution_context = neps_space.resolve(
                space, domain_sampler=sampling.OnlyPredefinedValuesSampler(predefined_samplings=samplings)
            )
            optimizer_creator_object = resolved_pipeline.optimizer_cls
            learning_rate = resolved_pipeline.learning_rate
            optimizer_creator = neps_space.convert_operation_to_callable(optimizer_creator_object)
            
            print(f"Learning rate: {learning_rate}")
            
            # Run NEPS RE optimizer
            def neps_opt_wrapper(params, lr, weight_decay=0.0):
                return optimizer_creator(params, lr=lr, variables=(1, 1))
            
            neps_creator = partial(neps_opt_wrapper)
            result = run_single_optimizer(
                opt_creator=neps_creator,
                learning_rate=learning_rate,
                optimizer_name='RE',
                benchmark_name=benchmark,
                n_steps=n_steps,
                seed=seed,
                n_samples=n_samples,
                n_dims=n_dims,
                weight_decay=weight_decay,
                output_dir=benchmark_dir
            )
            
            # Update trials for RE
            score = result['final_loss']
            new_trial = Trial(
                config=config.config, 
                metadata=trial.MetaData(
                    id=str(optimizer_idx), 
                    location="", 
                    state=None, 
                    previous_trial_id=None, 
                    previous_trial_location=None, 
                    sampling_worker_id="", 
                    time_sampled=optimizer_idx, 
                    time_end=optimizer_idx
                ), 
                report=trial.Report(
                    objective_to_minimize=score, 
                    cost=None, 
                    learning_curve=None, 
                    extra={}, 
                    err=None, 
                    tb=None, 
                    reported_as=trial.State.SUCCESS, 
                    evaluation_duration=0
                )
            )
            trials[str(optimizer_idx)] = new_trial

    print(f"\n{'='*60}")
    print(f"All results saved to {benchmark_dir}")
    print(f"{'='*60}")

