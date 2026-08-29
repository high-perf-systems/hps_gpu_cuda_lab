# ============================================================
# Experiment 5: 3D Stencil — torch.conv3d correctness reference
# hps_gpu_cuda_lab
#
# Correctness oracle ONLY (NOT a performance target — see
# problem.md Concept 6: conv3d pays for a dense (2r+1)^3 cube,
# (2r+1)^3 - 6r - 1 of which are multiplies against zero, so its
# timing is not a meaningful comparison point). Builds a dense
# kernel with zeros off the three axes, so F.conv3d computes
# exactly the same sparse stencil sum as your CUDA kernels.
#
# Do NOT add a performance sweep here. Timing/bandwidth/GFLOPS
# comparisons across kernel *versions* belong in stencil_harness.py,
# applied to your own launch_* functions — that's the actual
# question this experiment is trying to answer (see problem.md
# "Versions to Implement" and Q2-Q5). This file only answers
# "is the output correct," never "how fast was it."
#
# Usage on Colab:
#   paste this whole file into a cell (or %%writefile it and
#   `import stencil_reference as ref`), then:
#
#   out = ref.run_torch_reference(inp, radius=1)
#   ok  = torch.allclose(out, your_kernel_output, atol=1e-4)
#
# gpu_timer() and compute_metrics() below are generic utilities
# (not torch-specific) — stencil_harness.py imports them to time
# and score YOUR kernels against each other.
# ============================================================

import torch
import torch.nn.functional as F

T4_PEAK_BW_GBS = 320.0     # T4 memory bandwidth peak
T4_PEAK_GFLOPS = 8141.0    # T4 FP32 compute peak

# ------------------------------------------------------------
# Standard central-difference coefficients for the 1D second
# derivative (Fornberg), order 2/4/6 <-> radius 1/2/3.
# Indexed [0]=center, [1]=+-1, [2]=+-2, [3]=+-3.
# ------------------------------------------------------------
COEFFS_1D = {
    1: [-2.0, 1.0],                                  # 2nd order
    2: [-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0],          # 4th order
    3: [-49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0],  # 6th order
}


def build_coeffs_array(radius: int, dtype=torch.float32, device="cuda") -> torch.Tensor:
    """
    Single source of truth for stencil coefficients — shared by both
    build_stencil_kernel() (the dense conv3d reference, below) and the
    CUDA kernels' `coeffs` argument (see stencil_kernels.cu's
    launch_naive, which expects exactly this layout).

    Returns a 1D tensor of length radius+1:
        [0]    = center coefficient, summed over all 3 axes (3 * c0)
        [1..r] = neighbor coefficient at distance k (same value used
                 for all 3 axes, since the stencil is isotropic)
    """
    c = COEFFS_1D[radius]
    center = 3.0 * c[0]
    values = [center] + list(c[1:])
    return torch.tensor(values, dtype=dtype, device=device)


def build_stencil_kernel(radius: int, dtype=torch.float32, device="cuda"):
    """
    Build a dense (2r+1)^3 kernel tensor for use with F.conv3d that is
    exactly equivalent to the sparse axis-aligned stencil of the given
    radius. All entries are zero except the center and the six
    axis-aligned neighbor rays.

    Returns shape (1, 1, 2r+1, 2r+1, 2r+1) — ready for F.conv3d's
    (out_channels, in_channels, kD, kH, kW) weight layout.
    """
    coeffs = build_coeffs_array(radius, dtype=dtype, device=device)
    size = 2 * radius + 1
    mid = radius
    k = torch.zeros((size, size, size), dtype=dtype, device=device)

    k[mid, mid, mid] = coeffs[0]

    for offset in range(1, radius + 1):
        c = coeffs[offset]
        # z axis (dim 0), y axis (dim 1), x axis (dim 2)
        k[mid + offset, mid, mid] = c
        k[mid - offset, mid, mid] = c
        k[mid, mid + offset, mid] = c
        k[mid, mid - offset, mid] = c
        k[mid, mid, mid + offset] = c
        k[mid, mid, mid - offset] = c

    return k.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size, size)


def run_torch_reference(inp: torch.Tensor, radius: int) -> torch.Tensor:
    """
    inp: (D, H, W) float32 CUDA tensor.
    Returns (out_D, out_H, out_W) = (D-2r, H-2r, W-2r) — valid mode,
    matching the CUDA kernels' boundary convention (no padding).
    """
    kernel = build_stencil_kernel(radius, dtype=inp.dtype, device=inp.device)
    out = F.conv3d(
        inp.unsqueeze(0).unsqueeze(0),   # (1, 1, D, H, W)
        kernel,
        padding=0,
    )
    return out.squeeze(0).squeeze(0)


# ------------------------------------------------------------
# Timing
# ------------------------------------------------------------
def gpu_timer(fn, *args, warmup=3, runs=20):
    """
    Time a GPU function with CUDA events.
    warmup runs are discarded (primes L2, triggers clock boost).
    Returns average milliseconds over `runs` iterations.
    """
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


def compute_metrics(N: int, radius: int, time_ms: float):
    """
    Effective bandwidth for an N^3 cubic grid, valid-mode output.
    bytes_moved counts each input/output point once (see problem.md
    "Performance Metrics" — this is a *minimum bytes* convention,
    not what any particular kernel actually fetches from DRAM).

    flops assumes exactly 6*radius+1 taps per output point — true
    for your own kernels (they only ever touch the nonzero taps),
    but NOT true for run_torch_reference(), which computes the full
    dense (2r+1)^3 cube under the hood. Only call this on timings
    from your own launch_* functions, not on the torch reference.
    """
    out_n = N - 2 * radius
    in_pts = N ** 3
    out_pts = max(out_n, 0) ** 3
    bytes_moved = (in_pts + out_pts) * 4  # float32
    seconds = time_ms / 1000.0
    bw_gbs = bytes_moved / seconds / 1e9
    flops = 2 * (6 * radius + 1) * out_pts
    gflops = flops / seconds / 1e9
    return {
        "N": N,
        "radius": radius,
        "time_ms": time_ms,
        "bandwidth_GBs": bw_gbs,
        "pct_peak_bw": bw_gbs / T4_PEAK_BW_GBS * 100.0,
        "gflops": gflops,
    }


# ------------------------------------------------------------
# Self-test — confirms this file works standalone (shapes come
# out right, no NaNs). Not a benchmark; see stencil_harness.py
# for comparing your kernel versions against each other.
# ------------------------------------------------------------
if __name__ == "__main__":
    for radius in (1, 2, 3):
        N = 16
        inp = torch.randn(N, N, N, device="cuda", dtype=torch.float32)
        out = run_torch_reference(inp, radius)
        expected_shape = (N - 2 * radius,) * 3
        assert tuple(out.shape) == expected_shape, (out.shape, expected_shape)
        assert torch.isfinite(out).all()
        print(f"radius={radius}: output shape {tuple(out.shape)} OK")
