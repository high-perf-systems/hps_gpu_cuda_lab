# ============================================================
# Experiment 5: 3D Stencil — build + correctness harness
# hps_gpu_cuda_lab
#
# The ONLY file that touches the compiled CUDA extension. It:
#   1. Compiles stencil_kernels.cu via torch's JIT CUDA loader.
#   2. Registers each exposed launch_* function under a version
#      name (KERNELS).
#   3. Runs every registered version against the torch.conv3d
#      reference in stencil_reference.py and reports pass/fail.
#
# Usage on Colab:
#   Upload stencil_kernels.cu, stencil_reference.py, and this
#   file into /content/, then run this file (or paste it as a
#   notebook cell — it's plain top-to-bottom script, no __main__
#   guard needed for that use case, but one is included so it
#   also works as `python stencil_harness.py`).
# ============================================================

import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["TORCH_SHOW_CPP_EXTENSION_DEBUG"] = "1"

import torch
import pandas as pd
from torch.utils.cpp_extension import load

import stencil_reference as ref
from stencil_reference import gpu_timer, compute_metrics

# ------------------------------------------------------------
# COMPILE
#
# KNOWN GOTCHA while iterating on the kernel: CPython cannot
# cleanly reload a native extension module in a live session. If
# you edit stencil_kernels.cu and re-run this cell with the same
# module `name` in the same Colab runtime, you'll get:
#   ImportError: dynamic module does not define module export
#   function (PyInit_stencil_kernels)
# This is NOT a compile error — scroll up in the verbose output
# first to rule out a real nvcc/g++ error, then either:
#   (a) Runtime -> Restart session, then re-run this cell, or
#   (b) give the module a fresh name each rebuild (no restart
#       needed) via MODULE_NAME below, or
#   (c) `!rm -rf /content/cuda_build` if a stale cached .so from
#       an earlier broken attempt is being reused.
# ------------------------------------------------------------
import time

EXTRA_CUDA_CFLAGS = ["-gencode=arch=compute_75,code=sm_75"]  # T4 on Colab
EXTRA_LDFLAGS = ["-lcudart"]
BUILD_DIR = os.path.join(os.getcwd(), "cuda_build")
os.makedirs(BUILD_DIR, exist_ok=True)

# Versioned name sidesteps the reload gotcha above without needing
# a runtime restart on every iteration. Switch to a fixed name once
# the kernel is stable and you're no longer recompiling every run.
MODULE_NAME = f"stencil_kernels_{int(time.time())}"

stencil_module = load(
    name=MODULE_NAME,
    sources=["/content/stencil_kernels.cu"],
    extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
    extra_ldflags=EXTRA_LDFLAGS,
    build_directory=BUILD_DIR,
    verbose=True,
)

# ------------------------------------------------------------
# KERNEL REGISTRY
#
# One entry per launch_* function in stencil_kernels.cu (don't forget
# the matching m.def(...) line in that file's PYBIND11_MODULE block
# when adding a new one). KERNEL_SETUP holds any one-time, untimed
# setup call a version needs before it can be called repeatedly (e.g.
# copying coeffs into constant memory) — same split exp4's notebook
# used for its constant-memory version.
#
# NAMING NOTE: "4_register_tiled" is chronologically/conceptually
# Version 5 (see problem.md "Versions to Implement"). Thread
# coarsening (V4) was built and benchmarked FIRST as the initial
# attempt at the z-sweep; register tiling (V5) came after, once V4
# turned out to underperform even naive (see notes.md Finding 2/3
# for why). The "4_" prefix here is a harness-naming leftover, not a
# real version number.
# ------------------------------------------------------------
def run_naive(inp, coeffs, radius):
    return stencil_module.launch_naive(inp, coeffs, radius)


# launch_constmem / launch_smemtile / launch_threadcoarse /
# launch_registertile all read coefficients from constant memory
# rather than taking them as an argument — these adapters keep the
# common (inp, coeffs, radius) -> output interface that
# check_version/benchmark_version expect, and KERNEL_SETUP (below)
# is what actually copies `coeffs` into constant memory beforehand.
def run_constmem(inp, coeffs, radius):
    return stencil_module.launch_constmem(inp, radius)


def run_smemtile(inp, coeffs, radius):
    return stencil_module.launch_smemtile(inp, radius)


def run_threadcoarse(inp, coeffs, radius):
    return stencil_module.launch_threadcoarse(inp, radius)


def run_registertile(inp, coeffs, radius):
    return stencil_module.launch_registertile(inp, radius)


KERNELS = {
    "1_naive": run_naive,
    "2_constmem": run_constmem,
    "3_shared_tiled": run_smemtile,
    "4_thread_coarse": run_threadcoarse,
    "4_register_tiled": run_registertile,
}

KERNEL_SETUP = {
    # V2-V5 all read from d_coeffs — each needs this run before every
    # call (not just once), since coeffs differ across radii.
    "2_constmem": lambda inp, coeffs, radius: stencil_module.copyCoeffsToConstant(coeffs),
    "3_shared_tiled": lambda inp, coeffs, radius: stencil_module.copyCoeffsToConstant(coeffs),
    "4_thread_coarse": lambda inp, coeffs, radius: stencil_module.copyCoeffsToConstant(coeffs),
    "4_register_tiled": lambda inp, coeffs, radius: stencil_module.copyCoeffsToConstant(coeffs),
}

# ------------------------------------------------------------
# COMPARER
# ------------------------------------------------------------
def check_version(name, fn, inp, coeffs, radius, atol=1e-4):
    if name in KERNEL_SETUP:
        KERNEL_SETUP[name](inp, coeffs, radius)

    out = fn(inp, coeffs, radius)
    expected = ref.run_torch_reference(inp, radius)

    if out.shape != expected.shape:
        print(
            f"  [{name}] FAIL — shape mismatch: "
            f"got {tuple(out.shape)}, expected {tuple(expected.shape)}"
        )
        return False

    max_err = (out - expected).abs().max().item()
    ok = torch.allclose(out, expected, atol=atol)
    print(f"  [{name}] {'PASS' if ok else 'FAIL'}  max_err={max_err:.3e}")
    return ok


def check_all(N=64, radius=1, atol=1e-4):
    print(f"N={N}  radius={radius}")
    inp = torch.randn(N, N, N, device="cuda", dtype=torch.float32).contiguous()
    coeffs = ref.build_coeffs_array(radius)

    results = {}
    for name, fn in KERNELS.items():
        results[name] = check_version(name, fn, inp, coeffs, radius, atol=atol)
    return results


# ------------------------------------------------------------
# BENCHMARK — this is the actual comparison across kernel
# versions (problem.md "Performance Metrics" / Q2-Q5). Timing
# and bandwidth/GFLOPS are computed on YOUR launch_* functions,
# never on the torch reference (see stencil_reference.py header
# for why that comparison would be meaningless).
# ------------------------------------------------------------
def benchmark_version(name, fn, inp, coeffs, radius, N, warmup=3, runs=20):
    if name in KERNEL_SETUP:
        KERNEL_SETUP[name](inp, coeffs, radius)   # one-time, untimed

    def call():
        return fn(inp, coeffs, radius)

    time_ms = gpu_timer(call, warmup=warmup, runs=runs)
    metrics = compute_metrics(N, radius, time_ms)
    metrics["version"] = name
    print(
        f"  [{name}] time={time_ms:8.4f} ms  "
        f"BW={metrics['bandwidth_GBs']:7.1f} GB/s "
        f"({metrics['pct_peak_bw']:4.1f}% peak)  "
        f"{metrics['gflops']:7.1f} GFLOPS"
    )
    return metrics


def benchmark_all(N, radius, warmup=3, runs=20):
    print(f"N={N}  radius={radius}")
    inp = torch.randn(N, N, N, device="cuda", dtype=torch.float32).contiguous()
    coeffs = ref.build_coeffs_array(radius)

    rows = []
    for name, fn in KERNELS.items():
        rows.append(benchmark_version(name, fn, inp, coeffs, radius, N, warmup, runs))
    return rows


def benchmark_sweep(n_values, r_values, warmup=3, runs=20):
    """Prints a version x N comparison table for each radius — the
    'how do our kernels stack up against each other' view."""
    rows = []
    for N in n_values:
        for radius in r_values:
            rows.extend(benchmark_all(N, radius, warmup, runs))
            print()

    df = pd.DataFrame(rows)
    for radius in r_values:
        sub = df[df.radius == radius]
        print("\n" + "=" * 68)
        print(f"  Kernel Time (ms)  —  radius={radius}")
        print("=" * 68)
        print(sub.pivot(index="version", columns="N", values="time_ms").to_string())

        print(f"\n  Effective Bandwidth (GB/s)  —  radius={radius}  [T4 peak = 320]")
        print(sub.pivot(index="version", columns="N", values="bandwidth_GBs").to_string())

    return df


# ------------------------------------------------------------
# CORRECTNESS SWEEP — small sizes, every implemented radius.
# ------------------------------------------------------------
def correctness_sweep(n_values, r_values):
    all_ok = True
    for N in n_values:
        for radius in r_values:
            results = check_all(N=N, radius=radius)
            all_ok = all_ok and all(results.values())
            print()
    return all_ok


if __name__ == "__main__":
    CORRECTNESS_N = [32, 64, 128]
    BENCHMARK_N = [64, 128, 256, 384]   # matches problem.md "Input Sizes to Test"
    R_VALUES = [1, 2, 3]

    print("=" * 68)
    print("CORRECTNESS")
    print("=" * 68)
    all_ok = correctness_sweep(CORRECTNESS_N, R_VALUES)
    print("ALL PASS" if all_ok else "SOME FAILED — see above")

    if all_ok:
        print("\n" + "=" * 68)
        print("BENCHMARK")
        print("=" * 68)
        benchmark_sweep(BENCHMARK_N, R_VALUES)
    else:
        print("\nSkipping benchmark sweep — fix correctness failures first.")
