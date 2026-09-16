# ============================================================
# Experiment 6: Parallel Histogram — build + correctness + benchmark harness
# hps_gpu_cuda_lab
#
# The ONLY file that touches the compiled CUDA extension. It:
#   1. Compiles histogram.cu via torch's JIT CUDA loader.
#   2. Registers each exposed launch_* function under a version
#      name (KERNELS).
#   3. Checks every registered version against cpu_histogram() in
#      histogram_reference.py.
#   4. Times each version and sweeps N x num_bins x distribution,
#      tabulating runtime/throughput/bandwidth (problem.md
#      "Performance Metrics"), plus the contention-penalty ratio
#      time(skewed)/time(uniform) — the key number for this
#      experiment (problem.md Concept 2).
#
# Usage on Colab:
#   Upload histogram.cu, histogram_reference.py, and this file
#   into /content/, then run this file (or paste it as a notebook
#   cell — it's plain top-to-bottom script, no __main__ guard
#   needed for that use case, but one is included so it also
#   works as `python histogram_harness.py`).
# ============================================================

import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["TORCH_SHOW_CPP_EXTENSION_DEBUG"] = "1"

import time

import torch
import pandas as pd
from torch.utils.cpp_extension import load

import histogram_reference as ref

T4_PEAK_BW_GBS = 320.0  # T4 memory bandwidth peak


def gpu_timer(fn, *args, warmup=3, runs=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / runs


def compute_metrics(N, time_ms):
    seconds = time_ms / 1000.0
    bandwidth_GBs = (N * 4) / seconds / 1e9  # int32, 4 bytes/elem
    return {
        "N": N,
        "time_ms": time_ms,
        "throughput_elem_s": N / seconds,
        "bandwidth_GBs": bandwidth_GBs,
        "pct_peak_bw": bandwidth_GBs / T4_PEAK_BW_GBS * 100.0,
    }


# ------------------------------------------------------------
# COMPILE
#
# KNOWN GOTCHA while iterating on the kernel: CPython cannot cleanly
# reload a native extension module in a live session. If you edit
# histogram.cu and re-run this cell with the same module `name` in
# the same Colab runtime, you'll get:
#   ImportError: dynamic module does not define module export
#   function (PyInit_histogram...)
# This is NOT a compile error — scroll up in the verbose output
# first to rule out a real nvcc/g++ error, then either:
#   (a) Runtime -> Restart session, then re-run this cell, or
#   (b) give the module a fresh name each rebuild (no restart
#       needed) via MODULE_NAME below, or
#   (c) `!rm -rf /content/cuda_build` if a stale cached .so from
#       an earlier broken attempt is being reused.
# ------------------------------------------------------------
EXTRA_CUDA_CFLAGS = ["-gencode=arch=compute_75,code=sm_75"]  # T4 on Colab
EXTRA_LDFLAGS = ["-lcudart"]
BUILD_DIR = os.path.join(os.getcwd(), "cuda_build")
os.makedirs(BUILD_DIR, exist_ok=True)

MODULE_NAME = f"histogram_{int(time.time())}"

histogram_module = load(
    name=MODULE_NAME,
    sources=["/content/histogram.cu"],
    extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
    extra_ldflags=EXTRA_LDFLAGS,
    build_directory=BUILD_DIR,
    verbose=True,
)


# ------------------------------------------------------------
# KERNEL REGISTRY — one entry per launch_* function in histogram.cu
# (don't forget the matching m.def(...) line in that file's
# PYBIND11_MODULE block when adding a new version).
# ------------------------------------------------------------
def run_naive(data, num_bins):
    return histogram_module.launch_naive(data, num_bins)


def run_privatized(data, num_bins):
    return histogram_module.launch_privatized(data, num_bins)


def run_coarsening(data, num_bins):
    return histogram_module.launch_coarsening(data, num_bins)

def run_aggregation(data, num_bins):
    return histogram_module.launch_aggregation(data, num_bins)


KERNELS = {
    "1_naive": run_naive,
    "2_privatized": run_privatized,
    "3_coarsened": run_coarsening,
    "4_aggregation" : run_aggregation,
}

GENERATORS = {
    "uniform": ref.generate_uniform,
    "skewed": ref.generate_skewed,
}


# ------------------------------------------------------------
# COMPARER
# ------------------------------------------------------------
def check_version(name, fn, data, num_bins, cpu_result):
    out = fn(data, num_bins)
    return ref.verify(out, cpu_result, kernel_name=name)


def check_all(N, num_bins, distribution="uniform", seed=0):
    print(f"N={N}  num_bins={num_bins}  distribution={distribution}")
    data = GENERATORS[distribution](N, num_bins, seed=seed)
    cpu_result = ref.cpu_histogram(data, num_bins)

    results = {}
    for name, fn in KERNELS.items():
        results[name] = check_version(name, fn, data, num_bins, cpu_result)
    return results


def correctness_sweep(n_values, bins_values, distributions):
    all_ok = True
    for N in n_values:
        for num_bins in bins_values:
            for distribution in distributions:
                results = check_all(N, num_bins, distribution)
                all_ok = all_ok and all(results.values())
                print()
    return all_ok


# ------------------------------------------------------------
# BENCHMARK — the actual "how do our kernels stack up" comparison
# (problem.md "Performance Metrics" / Q1-Q3). Timed against
# real kernel calls only, never cpu_histogram (that's the CPU
# floor, reported separately via cpu_bincount_timer).
# ------------------------------------------------------------
def benchmark_version(name, fn, data, num_bins, N, warmup=3, runs=20):
    def call():
        return fn(data, num_bins)

    time_ms = gpu_timer(call, warmup=warmup, runs=runs)
    metrics = compute_metrics(N, time_ms)
    metrics["version"] = name
    print(
        f"  [{name}] time={time_ms:9.4f} ms  "
        f"throughput={metrics['throughput_elem_s']:.3e} elem/s  "
        f"BW={metrics['bandwidth_GBs']:7.2f} GB/s ({metrics['pct_peak_bw']:4.1f}% peak)"
    )
    return metrics


def benchmark_all(N, num_bins, distribution, warmup=3, runs=20, seed=0):
    print(f"N={N}  num_bins={num_bins}  distribution={distribution}")
    data = GENERATORS[distribution](N, num_bins, seed=seed)

    cpu_time_ms = ref.cpu_bincount_timer(data, num_bins, warmup=warmup, runs=runs)
    print(f"  [cpu_bincount] time={cpu_time_ms:9.4f} ms")

    rows = []
    for name, fn in KERNELS.items():
        row = benchmark_version(name, fn, data, num_bins, N, warmup, runs)
        row["num_bins"] = num_bins
        row["distribution"] = distribution
        row["cpu_time_ms"] = cpu_time_ms
        row["speedup_vs_cpu"] = cpu_time_ms / row["time_ms"]
        rows.append(row)
    return rows


def benchmark_sweep(n_values, bins_values, distributions, warmup=3, runs=20):
    """Prints version x N tables per (num_bins, distribution), then a
    contention-penalty table (time(skewed)/time(uniform)) — the key
    number for this experiment (problem.md Concept 2 / Q1)."""
    rows = []
    for N in n_values:
        for num_bins in bins_values:
            for distribution in distributions:
                rows.extend(benchmark_all(N, num_bins, distribution, warmup, runs))
                print()

    df = pd.DataFrame(rows)

    for num_bins in bins_values:
        sub = df[df.num_bins == num_bins]
        for metric, label in [
            ("time_ms", "Kernel Time (ms)"),
            ("bandwidth_GBs", "Effective Bandwidth (GB/s)  [T4 peak = 320]"),
            ("speedup_vs_cpu", "Speedup vs CPU (np.bincount)"),
        ]:
            print("\n" + "=" * 68)
            print(f"  {label}  —  num_bins={num_bins}")
            print("=" * 68)
            for distribution in distributions:
                d = sub[sub.distribution == distribution]
                print(f"\n  distribution={distribution}")
                print(d.pivot(index="version", columns="N", values=metric).to_string())

    if "uniform" in distributions and "skewed" in distributions:
        merged = df[df.distribution.isin(["uniform", "skewed"])].pivot_table(
            index=["version", "num_bins"], columns=["distribution", "N"], values="time_ms"
        )
        penalty = merged["skewed"] / merged["uniform"]
        print("\n" + "=" * 68)
        print("  Contention Penalty  =  time(skewed) / time(uniform)")
        print("=" * 68)
        print(penalty.to_string())

    return df


if __name__ == "__main__":
    CORRECTNESS_N = [1_024, 100_000]
    BENCHMARK_N = [1_000_000, 4_000_000, 16_000_000, 64_000_000]  # problem.md N sweep
    BINS_VALUES = [8, 32, 256]  # problem.md num_bins sweep
    DISTRIBUTIONS = ["uniform", "skewed"]

    print("=" * 68)
    print("CORRECTNESS")
    print("=" * 68)
    all_ok = correctness_sweep(CORRECTNESS_N, BINS_VALUES, DISTRIBUTIONS)
    print("ALL PASS" if all_ok else "SOME FAILED — see above")

    if all_ok:
        print("\n" + "=" * 68)
        print("BENCHMARK")
        print("=" * 68)
        benchmark_sweep(BENCHMARK_N, BINS_VALUES, DISTRIBUTIONS)
    else:
        print("\nSkipping benchmark sweep — fix correctness failures first.")
