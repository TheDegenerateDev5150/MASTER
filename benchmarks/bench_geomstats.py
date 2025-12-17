#!/usr/bin/env python3
import argparse
import csv
import gc
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

TOOLKIT_NAME = "geomstats"

SUPPORTED_BACKENDS = ["numpy", "autograd", "pytorch"]
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


def setup_backend(backend_name):
    os.environ["GEOMSTATS_BACKEND"] = backend_name

    if "geomstats" in sys.modules:
        mods_to_remove = [k for k in sys.modules if k.startswith("geomstats")]
        for mod in mods_to_remove:
            del sys.modules[mod]

    import geomstats.backend as gs
    return gs


def get_manifold_module():
    from geomstats.geometry.euclidean import Euclidean
    from geomstats.geometry.hypersphere import Hypersphere
    from geomstats.geometry.poincare_ball import PoincareBall
    from geomstats.geometry.special_orthogonal import SpecialOrthogonal
    from geomstats.geometry.spd_matrices import SPDMatrices

    return {
        "Euclidean": Euclidean,
        "Hypersphere": Hypersphere,
        "PoincareBall": PoincareBall,
        "SpecialOrthogonal": SpecialOrthogonal,
        "SPDMatrices": SPDMatrices,
    }


def create_manifold(manifold_name, dim):
    classes = get_manifold_module()

    if manifold_name == "euclidean":
        return classes["Euclidean"](dim=dim)
    elif manifold_name == "sphere":
        return classes["Hypersphere"](dim=dim)
    elif manifold_name == "poincare":
        return classes["PoincareBall"](dim=dim)
    elif manifold_name == "so_n":
        return classes["SpecialOrthogonal"](n=dim)
    elif manifold_name == "spd":
        return classes["SPDMatrices"](n=dim)
    else:
        raise ValueError(f"Unknown manifold: {manifold_name}")


def generate_point_pairs(manifold, n_samples):
    points_a = manifold.random_point(n_samples)
    points_b = manifold.random_point(n_samples)
    return points_a, points_b


def generate_tangent_data(manifold, n_samples):
    base_points = manifold.random_point(n_samples)
    tangent_vecs = manifold.random_tangent_vec(base_points, n_samples)
    return base_points, tangent_vecs


def force_computation(result, gs):
    if hasattr(result, "numpy"):
        return float(result.sum().numpy())
    elif hasattr(result, "sum"):
        val = gs.sum(result)
        if hasattr(val, "item"):
            return val.item()
        return float(val)
    return float(result)


def run_dist(manifold, points_a, points_b, gs):
    results = []
    n = points_a.shape[0] if hasattr(points_a, "shape") and len(points_a.shape) > len(manifold.shape) else 1

    if n > 1:
        for i in range(n):
            r = manifold.metric.dist(points_a[i], points_b[i])
            results.append(r)
    else:
        r = manifold.metric.dist(points_a, points_b)
        results.append(r)

    return results


def run_exp(manifold, base_points, tangent_vecs, gs):
    results = []
    n = base_points.shape[0] if hasattr(base_points, "shape") and len(base_points.shape) > len(manifold.shape) else 1

    if n > 1:
        for i in range(n):
            r = manifold.metric.exp(tangent_vecs[i], base_points[i])
            results.append(r)
    else:
        r = manifold.metric.exp(tangent_vecs, base_points)
        results.append(r)

    return results


def run_log(manifold, points_a, points_b, gs):
    results = []
    n = points_a.shape[0] if hasattr(points_a, "shape") and len(points_a.shape) > len(manifold.shape) else 1

    if n > 1:
        for i in range(n):
            r = manifold.metric.log(points_b[i], points_a[i])
            results.append(r)
    else:
        r = manifold.metric.log(points_b, points_a)
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
    gs,
):
    if op_name == "dist":
        points_a, points_b = generate_point_pairs(manifold, n_samples)
        run_fn = lambda: run_dist(manifold, points_a, points_b, gs)
    elif op_name == "exp":
        base_points, tangent_vecs = generate_tangent_data(manifold, n_samples)
        run_fn = lambda: run_exp(manifold, base_points, tangent_vecs, gs)
    elif op_name == "log":
        points_a, points_b = generate_point_pairs(manifold, n_samples)
        run_fn = lambda: run_log(manifold, points_a, points_b, gs)
    else:
        raise ValueError(f"Unknown operation: {op_name}")

    for _ in range(n_warmup):
        results = run_fn()
        for r in results:
            force_computation(r, gs)

    times = []
    for _ in range(n_iterations):
        gc.collect()

        start = time.perf_counter()
        results = run_fn()
        for r in results:
            force_computation(r, gs)
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

    backends = args.backends if args.backends else SUPPORTED_BACKENDS
    manifolds = args.manifolds if args.manifolds else SUPPORTED_MANIFOLDS
    operations = args.operations if args.operations else SUPPORTED_OPERATIONS

    total_configs = 0
    for backend in backends:
        for manifold_name in manifolds:
            dims = get_dimensions(manifold_name, args)
            total_configs += len(dims) * len(operations)

    current = 0

    for backend in backends:
        print(f"\n=== Backend: {backend} ===", file=sys.stderr)

        try:
            gs = setup_backend(backend)
        except Exception as e:
            print(f"Failed to setup backend {backend}: {e}", file=sys.stderr)
            continue

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
                        f"{progress} {backend}/{manifold_name}/{op_name} dim={dim}",
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
                            gs=gs,
                        )

                        results.append({
                            "toolkit": TOOLKIT_NAME,
                            "backend": backend,
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
        description="Benchmark Geomstats manifold operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        choices=SUPPORTED_BACKENDS,
        default=None,
        help="Backends to benchmark (default: all)",
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
        default="results/geomstats_results.csv",
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

    print("Geomstats Benchmark", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Backends: {args.backends or SUPPORTED_BACKENDS}", file=sys.stderr)
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
