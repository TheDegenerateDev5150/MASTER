#!/usr/bin/env python3
import argparse
import csv
import gc
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

TOOLKIT_NAME = "tf_riemopt"
BACKEND_NAME = "tensorflow"

SUPPORTED_MANIFOLDS = ["euclidean", "sphere", "poincare", "so_n", "spd"]
SUPPORTED_OPERATIONS = ["dist", "exp", "log"]

DIMS_VECTOR = [10, 100, 1000, 10000, 100000, 1000000]
DIMS_MATRIX = [2, 4, 8, 16, 32]

CSV_FIELDNAMES = [
    "toolkit",
    "backend",
    "manifold",
    "operation",
    "dimension",
    "n_samples",
    "n_iterations",
    "mean_time_sec",
    "std_time_sec",
    "min_time_sec",
    "max_time_sec",
    "timestamp",
]


def get_manifold_classes():
    from tensorflow_riemopt.manifolds import (
        Euclidean,
        Sphere,
        Poincare,
        SpecialOrthogonal,
        SPDAffineInvariant,
    )

    return {
        "Euclidean": Euclidean,
        "Sphere": Sphere,
        "Poincare": Poincare,
        "SpecialOrthogonal": SpecialOrthogonal,
        "SPDAffineInvariant": SPDAffineInvariant,
    }


def create_manifold(manifold_name, dim):
    classes = get_manifold_classes()

    if manifold_name == "euclidean":
        return classes["Euclidean"]()
    elif manifold_name == "sphere":
        return classes["Sphere"]()
    elif manifold_name == "poincare":
        return classes["Poincare"](k=1.0)
    elif manifold_name == "so_n":
        return classes["SpecialOrthogonal"]()
    elif manifold_name == "spd":
        return classes["SPDAffineInvariant"]()
    else:
        raise ValueError(f"Unknown manifold: {manifold_name}")


def get_point_shape(manifold_name, dim):
    if manifold_name in ["euclidean", "poincare"]:
        return (dim,)
    elif manifold_name == "sphere":
        return (dim + 1,)
    elif manifold_name in ["so_n", "spd"]:
        return (dim, dim)
    else:
        raise ValueError(f"Unknown manifold: {manifold_name}")


def generate_random_point(manifold, manifold_name, dim, dtype):
    shape = get_point_shape(manifold_name, dim)

    if manifold_name == "euclidean":
        return tf.random.normal(shape, dtype=dtype)
    elif manifold_name == "sphere":
        raw = tf.random.normal(shape, dtype=dtype)
        return manifold.projx(raw)
    elif manifold_name == "poincare":
        raw = tf.random.uniform(shape, minval=-0.5, maxval=0.5, dtype=dtype)
        return raw
    elif manifold_name == "so_n":
        raw = tf.random.normal(shape, dtype=dtype)
        return manifold.projx(raw)
    elif manifold_name == "spd":
        raw = tf.random.normal(shape, dtype=dtype)
        return raw @ tf.transpose(raw) + tf.eye(dim, dtype=dtype)
    else:
        raise ValueError(f"Unknown manifold: {manifold_name}")


def generate_random_tangent(manifold, manifold_name, base_point, dtype):
    shape = base_point.shape
    raw = tf.random.normal(shape, dtype=dtype) * 0.1
    return manifold.proju(base_point, raw)


def generate_point_pairs(manifold, manifold_name, dim, n_samples, dtype):
    points_a = [generate_random_point(manifold, manifold_name, dim, dtype)
                for _ in range(n_samples)]
    points_b = [generate_random_point(manifold, manifold_name, dim, dtype)
                for _ in range(n_samples)]
    return points_a, points_b


def generate_tangent_data(manifold, manifold_name, dim, n_samples, dtype):
    base_points = [generate_random_point(manifold, manifold_name, dim, dtype)
                   for _ in range(n_samples)]
    tangent_vecs = [generate_random_tangent(manifold, manifold_name, bp, dtype)
                    for bp in base_points]
    return base_points, tangent_vecs


def force_computation(result):
    return float(tf.reduce_sum(result).numpy())


def run_dist(manifold, points_a, points_b):
    results = []
    for pa, pb in zip(points_a, points_b):
        r = manifold.dist(pa, pb)
        results.append(r)
    return results


def run_exp(manifold, base_points, tangent_vecs):
    results = []
    for bp, tv in zip(base_points, tangent_vecs):
        r = manifold.exp(bp, tv)
        results.append(r)
    return results


def run_log(manifold, points_a, points_b):
    results = []
    for pa, pb in zip(points_a, points_b):
        r = manifold.log(pa, pb)
        results.append(r)
    return results


def benchmark_operation(
    manifold,
    manifold_name,
    op_name,
    dim,
    n_samples,
    n_iterations,
    n_warmup,
    dtype,
):
    if op_name == "dist":
        points_a, points_b = generate_point_pairs(
            manifold, manifold_name, dim, n_samples, dtype
        )
        run_fn = lambda: run_dist(manifold, points_a, points_b)
    elif op_name == "exp":
        base_points, tangent_vecs = generate_tangent_data(
            manifold, manifold_name, dim, n_samples, dtype
        )
        run_fn = lambda: run_exp(manifold, base_points, tangent_vecs)
    elif op_name == "log":
        points_a, points_b = generate_point_pairs(
            manifold, manifold_name, dim, n_samples, dtype
        )
        run_fn = lambda: run_log(manifold, points_a, points_b)
    else:
        raise ValueError(f"Unknown operation: {op_name}")

    for _ in range(n_warmup):
        results = run_fn()
        for r in results:
            force_computation(r)

    times = []
    for _ in range(n_iterations):
        gc.collect()

        start = time.perf_counter()
        results = run_fn()
        for r in results:
            force_computation(r)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
    }


def get_dimensions(manifold_name, args):
    if manifold_name in ["so_n", "spd"]:
        if args.dims_matrix:
            return args.dims_matrix
        return DIMS_MATRIX
    else:
        if args.dims_vector:
            return args.dims_vector
        return DIMS_VECTOR


def run_benchmarks(args):
    results = []

    dtype = tf.float64

    manifolds = args.manifolds if args.manifolds else SUPPORTED_MANIFOLDS
    operations = args.operations if args.operations else SUPPORTED_OPERATIONS

    total_configs = 0
    for manifold_name in manifolds:
        dims = get_dimensions(manifold_name, args)
        total_configs += len(dims) * len(operations)

    current = 0

    for manifold_name in manifolds:
        dims = get_dimensions(manifold_name, args)

        for dim in dims:
            try:
                manifold = create_manifold(manifold_name, dim)
            except Exception as e:
                print(f"Failed to create {manifold_name}(dim={dim}): {e}", file=sys.stderr)
                current += len(operations)
                continue

            for op_name in operations:
                current += 1
                progress = f"[{current}/{total_configs}]"
                print(
                    f"{progress} {BACKEND_NAME}/{manifold_name}/{op_name} dim={dim}",
                    file=sys.stderr,
                )

                try:
                    stats = benchmark_operation(
                        manifold=manifold,
                        manifold_name=manifold_name,
                        op_name=op_name,
                        dim=dim,
                        n_samples=args.n_samples,
                        n_iterations=args.n_iterations,
                        n_warmup=args.n_warmup,
                        dtype=dtype,
                    )

                    results.append({
                        "toolkit": TOOLKIT_NAME,
                        "backend": BACKEND_NAME,
                        "manifold": manifold_name,
                        "operation": op_name,
                        "dimension": dim,
                        "n_samples": args.n_samples,
                        "n_iterations": args.n_iterations,
                        "mean_time_sec": stats["mean"],
                        "std_time_sec": stats["std"],
                        "min_time_sec": stats["min"],
                        "max_time_sec": stats["max"],
                        "timestamp": datetime.now().isoformat(),
                    })

                    print(
                        f"         mean={stats['mean']:.6f}s std={stats['std']:.6f}s",
                        file=sys.stderr,
                    )

                except Exception as e:
                    print(f"         FAILED: {e}", file=sys.stderr)
                    continue

            gc.collect()

    return results


def write_results(results, output_path, append=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append and output_path.exists() else "w"
    write_header = mode == "w" or not output_path.exists()

    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_path}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TensorFlow-RiemOpt manifold operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--manifolds",
        nargs="+",
        choices=SUPPORTED_MANIFOLDS,
        default=None,
        help="Manifolds to benchmark (default: all)",
    )

    parser.add_argument(
        "--operations",
        nargs="+",
        choices=SUPPORTED_OPERATIONS,
        default=None,
        help="Operations to benchmark (default: all)",
    )

    parser.add_argument(
        "--dims-vector",
        nargs="+",
        type=int,
        default=None,
        help="Dimensions for vector manifolds (euclidean, sphere, poincare)",
    )

    parser.add_argument(
        "--dims-matrix",
        nargs="+",
        type=int,
        default=None,
        help="Dimensions for matrix manifolds (so_n, spd)",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of random samples per benchmark",
    )

    parser.add_argument(
        "--n-iterations",
        type=int,
        default=10,
        help="Number of timed iterations",
    )

    parser.add_argument(
        "--n-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/tf_riemopt_results.csv",
        help="Output CSV file path",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("TensorFlow-RiemOpt Benchmark", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Backend: {BACKEND_NAME}", file=sys.stderr)
    print(f"TensorFlow version: {tf.__version__}", file=sys.stderr)
    print(f"Manifolds: {args.manifolds or SUPPORTED_MANIFOLDS}", file=sys.stderr)
    print(f"Operations: {args.operations or SUPPORTED_OPERATIONS}", file=sys.stderr)
    print(f"Samples: {args.n_samples}", file=sys.stderr)
    print(f"Iterations: {args.n_iterations}", file=sys.stderr)
    print(f"Warmup: {args.n_warmup}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)

    results = run_benchmarks(args)

    if results:
        write_results(results, args.output, append=args.append)
        print(f"\nCompleted {len(results)} benchmarks", file=sys.stderr)
    else:
        print("\nNo results to write", file=sys.stderr)


if __name__ == "__main__":
    main()
