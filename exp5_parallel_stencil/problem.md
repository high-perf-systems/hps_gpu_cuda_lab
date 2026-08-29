# Experiment 5: 3D Stencil (Finite Difference)

## Problem Statement

Given a 3D scalar grid `U` of size `D×H×W`, compute an output grid `O` where
each output point is a fixed linear combination of the input point and its
neighbors along each of the three axes:

```
O[d][r][c] = c0 * U[d][r][c]
           + Σ_k  cx[k] * U[d][r][c+k]         (k = -radius .. radius, k != 0)
           + Σ_k  cy[k] * U[d][r+k][c]
           + Σ_k  cz[k] * U[d+k][r][c]
```

This is a **finite-difference stencil** — it approximates a derivative (here,
the 3D Laplacian, ∇²U) at each grid point using nearby samples. The number of
neighbor points needed along each axis is the **order** of the stencil: a
2nd-order-accurate approximation needs 1 neighbor on each side per axis
(radius=1), 4th-order needs 2 (radius=2), 6th-order needs 3 (radius=3). Higher
order = wider stencil = more accurate derivative estimate = more points to
read per output.

With zero-padding *not* applied (valid mode, matching exp4's convention), the
output shrinks by `2*radius` in each dimension:

```
out_D = D - 2*radius,  out_H = H - 2*radius,  out_W = W - 2*radius
```

---

## Connection to Previous Experiments

| Experiment | Operation | Independent outputs? | Memory pattern | Bottleneck |
|------------|-----------|----------------------|-----------------|------------|
| Exp 1: Vec add | C[i] = A[i]+B[i] | yes | 1 read → 1 write | Bandwidth |
| Exp 2: Matmul | C[r][c] = dot(A_row, B_col) | yes | each input reused N times | Compute (large N) |
| Exp 3: Reduction | result = Σ A[i] | no — must communicate | tree of writes | Bandwidth |
| Exp 4: Conv2D | O[r][c] = Σ over K×K patch | yes | each input reused up to K² times | Bandwidth (small K) |
| **Exp 5: Stencil3D** | **O[d][r][c] = Σ over axis-aligned neighbors** | **yes** | **each input reused up to 6·radius+1 times, along axes only** | **Bandwidth — more so than conv2D** |

The key structural difference from exp4: a convolution kernel is a **dense**
K×K (or K×K×K) cube of independent, generally nonzero weights. A stencil is
**sparse** — only the center and the axis-aligned neighbors are nonzero; the
K×K×K cube's corners and edges (everywhere off the three axes) are exactly
zero. A radius-r stencil touches `6r+1` points, while a dense (2r+1)³ cube
would touch `(2r+1)³` points — e.g. at r=2, that's 13 vs 125. This sparsity
is what makes stencils dramatically more memory-bound than a same-radius
dense convolution: far fewer FLOPs are extracted per byte moved.

---

## Arithmetic Intensity

Following the roofline framework from exp2-4:

```
FLOPs per output point         ≈  2 × (6·radius + 1)   (1 multiply + 1 add per tap)
Minimum bytes moved (tiled,
  each input point loaded once) ≈  8 bytes/point  (4 read + 4 write, amortized)

Arithmetic intensity (tiled)   ≈  2×(6·radius+1) / 8   FLOP/byte

T4 ridge point                 =  8141 GFLOPS / 320 GB/s  =  25.4 FLOP/byte

radius=1 (7-point,  2nd order):  AI ≈ 1.75 FLOP/byte  → deep memory-bound
radius=2 (13-point, 4th order):  AI ≈ 3.25 FLOP/byte  → memory-bound
radius=3 (19-point, 6th order):  AI ≈ 4.75 FLOP/byte  → memory-bound
```

Every order tested here sits far below the T4's ridge point — unlike matmul
(exp2), no amount of tiling will make this kernel compute-bound. The goal is
purely to minimize redundant DRAM traffic, same objective as exp1 and exp3.

**Naive kernel (no reuse across threads)** reads all `6r+1` neighbors fresh
from global memory per output point:

```
AI (naive) ≈ 2×(6r+1) / (4×(6r+1) + 4)  ≈  0.5 FLOP/byte  (all radii)
```

This is worse than exp4's naive conv2D because a stencil has fewer FLOPs per
tap to amortize the same redundant fetch — tiling should matter *more* here,
not less.

---

## Core Concepts

### Concept 1: Order vs Radius — Standard Finite-Difference Coefficients

The "order" of the stencil is the accuracy order of the derivative it
approximates, which determines the radius (points needed per side per axis).
These are the standard central-difference coefficients for the 1D second
derivative (Fornberg's method), applied identically along x, y, and z, then
summed (the center coefficient accumulates 3× since it appears in all three
1D stencils):

| Order | Radius | 1D coefficients (center, ±1, ±2, ±3) | 3D stencil size |
|-------|--------|----------------------------------------|------------------|
| 2nd | 1 | `[-2, 1]` → center = 3×(-2) = -6, neighbors = 1 | 7-point |
| 4th | 2 | `[-5/2, 4/3, -1/12]` → center = 3×(-5/2) = -15/2 | 13-point |
| 6th | 3 | `[-49/18, 3/2, -3/20, 1/90]` → center = 3×(-49/18) = -49/6 | 19-point |

These are fixed, well-known coefficients (not learned) — the same set is
used by every kernel version and by the reference implementation, so there
is no "training" step; this experiment is purely about the memory/compute
mechanics of applying a known stencil fast.

### Concept 2: The Halo Problem Gets Bigger in 3D

Exp4 introduced the halo (apron) concept for 2D tiles: a `TILE×TILE` block of
output threads needs a `(TILE+2r)×(TILE+2r)` input tile in shared memory. In
3D, a naive extension needs a `(TILE+2r)³` shared-memory cube. This grows
**cubically** with tile size and radius:

```
TILE=8,  r=1: (8+2)³  × 4 bytes =  4,000 bytes  ≈ 3.9 KB   — fine
TILE=8,  r=3: (8+6)³  × 4 bytes = 10,976 bytes  ≈ 10.7 KB  — still fine
TILE=16, r=1: (16+2)³ × 4 bytes = 23,328 bytes  ≈ 22.8 KB  — eating into 64KB budget
TILE=16, r=3: (16+6)³ × 4 bytes = 42,592 bytes  ≈ 41.6 KB  — most of the SM's shared memory for ONE block
```

A full 3D shared-memory tile quickly limits occupancy (few blocks per SM fit
in 64 KB) — this motivates the standard technique covered in the PMPP
stencil chapter:

### Concept 3: Register Tiling — the Z-Sweep Pattern

Instead of loading a full 3D tile into shared memory, keep only a **2D x-y
plane tile** in shared memory (same cost as exp4's 2D tiling) and stream
through the z dimension one plane at a time, holding the `2·radius+1`
z-neighbor planes needed for the current output plane in **registers** per
thread (a small local array, e.g. `float in_front[radius]`, `float
in_back[radius]`, and one running "current" value). Each thread:

1. Loads its (x,y) column of `2r+1` z-values into registers once.
2. Slides the register window forward one z-plane at a time, loading only
   the *new* leading plane's value from shared memory into the corresponding
   thread's register and reusing the rest.
3. Computes the axis-x and axis-y contributions from the shared-memory tile
   (with 2D halo) and the axis-z contribution from its own register window.

This trades 3D shared-memory cost for a small, fixed number of registers per
thread — shared memory usage stays `O((TILE+2r)²)` instead of
`O((TILE+2r)³)`, at the cost of `O(radius)` extra registers per thread and
some redundant loading of the 2D plane's halo at each z-step.

### Concept 4: Thread Coarsening

Since each thread's per-point FLOP count is small (a handful of FMAs) relative
to the overhead of computing indices and checking bounds, having each thread
compute multiple output points (e.g., a short run along z) amortizes that
fixed overhead and increases the register-tiling reuse window without
proportionally increasing shared memory traffic.

### Concept 5: Boundary Handling — Valid Mode

No padding is applied. Threads whose stencil would read outside
`[0,D)×[0,H)×[0,W)` are not launched at all — the output grid is smaller than
the input by `2·radius` in each dimension, exactly like exp4's `out_H =
H-Kh+1`. This avoids branching for the padding case inside the hot loop.

### Concept 6: Reference Implementation — No "cuDNN of Stencils"

Unlike exp4, where `torch.nn.functional.conv2d` (cuDNN) served as both a
correctness reference *and* a meaningful performance ceiling, there is no
equally dominant, heavily-tuned vendor library for finite-difference stencils
running on GPUs. Options considered and why they were set aside:

- **`cupyx.scipy.ndimage`** — correct, GPU-resident, but not aggressively
  tuned for tiny fixed-radius stencils; not a meaningful perf ceiling.
- **Devito** — a real finite-difference DSL used in seismic imaging that
  auto-generates optimized CUDA stencil kernels; the closest thing to a
  "cuDNN of stencils," but a heavy dependency out of scope for this
  experiment.
- **`torch.nn.functional.conv3d`** — chosen here, for **correctness only**.
  A stencil is a convolution with a sparse, fixed kernel, so it can be
  expressed exactly as a dense `(2r+1)³` kernel with zeros everywhere off
  the three axes. `conv3d` (cuDNN) computes this correctly, but it is
  deliberately *not* used as a performance target: cuDNN pays for the full
  dense cube (`(2r+1)³` multiplies) even though `(2r+1)³ - 6r - 1` of those
  multiplies are against zero. Our kernels, by only ever looping over the
  `6r+1` nonzero taps, do strictly less work than the reference by
  construction — a favorable comparison is not the point.

This experiment therefore benchmarks stencil kernel **versions against each
other** (naive vs constant-memory vs shared-tiled vs register-tiled), using
`conv3d` solely as the correctness oracle, not a performance target.

---

## Implementation Notes

Following exp4's pattern, kernels are written in CUDA C in `stencil_kernels.cu`
and compiled from a Google Colab notebook using
`torch.utils.cpp_extension.load` (nvcc, T4 GPU). The Python harness calls the
compiled functions directly, times them with `torch.cuda.Event`, and compares
against `F.conv3d` with a dense zero-padded kernel built from the coefficient
tables above.

No CPU baseline is implemented, for the same reason as exp4: a Python/NumPy
triple-nested-loop reference would be orders of magnitude slower and is not a
useful baseline for optimization decisions.

---

## Versions to Implement

### Version 1: GPU Naive (Global Memory Only)
One thread per output point. Each thread reads its `6r+1` neighbors directly
from global memory and its coefficients from a small array passed as a
kernel argument. No shared memory, no constant memory.

Purpose: raw throughput baseline; expected to be badly memory-bound due to
redundant re-fetching of the same input points by neighboring threads.

### Version 2: GPU with Constant Memory for Coefficients
Same structure as Version 1, but the `6r+1` (or `2·radius+1` per axis, since
they're reused across axes for an isotropic stencil) coefficients are stored
in `__constant__` memory via `cudaMemcpyToSymbol`, broadcasting to every
thread in a warp in one cycle instead of a per-thread global read.

Purpose: isolate the (likely small, since coefficient count is tiny) benefit
of constant memory, same rationale as exp4 Version 2.

### Version 3: GPU Shared-Memory Tiling (Full 3D Tile + Halo)
Thread blocks collaboratively load a `(TILE+2r)³` input tile (including the
3D halo) into shared memory before computing. Direct 3D extension of exp4's
2D tiling.

Purpose: demonstrate the shared-memory blow-up described in Concept 2 —
expect this version to hit occupancy limits at larger tile sizes/radii, and
to motivate Version 4.

### Version 4: GPU Thread-Coarsened Z-Sweep (rolling shared-memory planes)
A block covers a 2D x-y output tile and coarsens across z: instead of one
thread per output point, each thread sweeps a chunk of z-depths in a loop,
keeping a rolling window of `2·radius+1` *fully haloed* x-y planes in shared
memory (drop the oldest plane, load one new leading plane, per z-step).

Purpose: first attempt at Concept 4 (thread coarsening) — grid must still be
chunked in z (`gridDim.z = ceil(oD/Z_CHUNK)`), not swept in a single block,
or occupancy collapses (see notes.md Finding 3). Turned out to be a useful
intermediate step, not the final design — see Version 5.

### Version 5: GPU Register-Tiled (Z-Sweep)
Same block/coarsening structure as Version 4, but only the **center** x-y
plane is kept in shared memory (with its halo, for the x/y taps). The
`2·radius` z-neighbor taps never need an x/y halo at all — the stencil's
axis-aligned cross shape means a z-tap always reads the exact same (row,
col) as the center, never an offset one — so they're kept as a small
per-thread register array instead (Concept 3), templated on `radius` so the
compiler can unroll the tap loop and keep that array in real registers
rather than spilling it to local memory.

Purpose: primary optimization target of this experiment — quantify how much
avoiding the full 3D (and even the full rolling-multi-plane) shared-memory
footprint actually buys, and find whether register tiling beats plain 3D
shared tiling (V3) and naive/const-mem (V1/V2) at each radius.

---

## Input Sizes to Test

Grids are cubic, `N×N×N`. Memory scales as `O(N³)`, so sizes are kept modest
to stay well within the T4's 15.6 GB and to keep sweep time reasonable:

| N | Grid | Input bytes (float32) |
|---|------|------------------------|
| 64  | 64³  | 1.0 MB |
| 128 | 128³ | 8.4 MB |
| 256 | 256³ | 67.1 MB |
| 384 | 384³ | 226.5 MB |

**Radius sweep** (fixed N=256): `radius ∈ {1, 2, 3}` — 2nd, 4th, 6th order.

---

## Performance Metrics

**Primary metric: effective memory bandwidth (GB/s)**

```
bytes_moved = (D×H×W + out_D×out_H×out_W) × 4
bandwidth   = bytes_moved / kernel_time_seconds
```

Same "minimum bytes" convention as exp4 — counts each input/output point
once regardless of how many times a kernel actually re-fetches it from DRAM.

**Secondary metric: kernel time (ms)**, fixed N=256, radius=1, across versions.

**Tertiary: % of T4 peak bandwidth (320 GB/s)**.

---

## Timing Methodology

Identical to exp4 — `torch.cuda.Event`, 3 warmup runs discarded, 20 timed
runs averaged:

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

for _ in range(3):           # warmup
    run_kernel(...)
torch.cuda.synchronize()

start.record()
for _ in range(20):
    run_kernel(...)
end.record()
torch.cuda.synchronize()

ms_per_run = start.elapsed_time(end) / 20
```

---

## Hypothesis

### Q1: How much does constant memory help over naive?
Prediction: small, similar to exp4 — the coefficient array is tiny (at most
19 floats) and likely already L2-resident after the first few warps even in
Version 1. Expect < 10% speedup.

### Q2: How much does shared-memory tiling (V3) help over naive/const-mem?
Prediction: large — naive re-fetches each input point up to `6r+1` times from
DRAM; tiling should cut redundant DRAM traffic dramatically. Expect
2–6× speedup at radius=1, growing with radius (more reuse potential per
loaded point, same reasoning as exp4 Q3/Q6).

### Q3: Does full 3D shared tiling (V3) run into the occupancy wall predicted
in Concept 2?
Prediction: yes, especially at radius=3 with larger tile sizes (e.g. TILE=16)
— expect measurably reduced occupancy (fewer blocks/SM) or a forced drop to
smaller tile sizes to fit the 64 KB shared memory budget.

### Q4: Does register tiling (V5) beat full 3D shared tiling (V3)?
Prediction: V5 wins at larger radii, where V3's cubic shared-memory cost
becomes the bottleneck; V3 may be competitive or even slightly faster at
radius=1, where the 3D tile is still small and V5's extra per-thread register
pressure and repeated 2D-plane halo loading eat into the benefit.
(Note: V4, the thread-coarsened rolling-plane design, was built and
benchmarked first, as a naive attempt at coarsening — see notes.md for why
it underperforms both V3 and V5, and why V5 was needed on top of it.)

### Q5: Does the tiled-vs-naive speedup grow with radius, as in exp4?
Prediction: yes — arithmetic reuse per input point grows with radius (more
output points share each loaded input point), so tiling should help more at
radius=3 than radius=1, mirroring exp4's K-scaling result.

### Q6: How does effective bandwidth compare to exp4's conv2D at equivalent
radius?
Prediction: worse — a stencil moves the same halo-tile bytes as a dense
convolution of the same radius but extracts far fewer FLOPs (linear in
radius vs cubic), so it should saturate at a lower fraction of peak
bandwidth per unit of *useful compute*, even if raw GB/s achieved is similar.

---

## Forward — Possible Experiment 6

The PMPP stencil chapter's remaining techniques not covered here — thread
coarsening tuned per-radius, and multi-GPU halo exchange (splitting a large
3D grid across GPUs and exchanging boundary planes) — are deferred to a
possible Experiment 6, after evaluating whether register tiling (V4) here
already closes most of the gap to the bandwidth roofline.

---

## What Success Looks Like

After this experiment I should be able to:
    ✅ Implement a 3D axis-aligned stencil kernel (naive) confidently
    ✅ Explain why a stencil is more memory-bound than a same-radius dense convolution
    ✅ Calculate 3D shared-memory tile cost including halo, and explain why it grows cubically
    ✅ Implement register-tiled z-sweep stencil without correctness bugs
    ✅ Explain the register-vs-shared-memory tradeoff for 3D neighborhoods
    ✅ Measure effective bandwidth and compare across stencil orders (radius 1/2/3)
    ✅ Explain why torch.conv3d/cuDNN is a correctness oracle here, not a performance ceiling
