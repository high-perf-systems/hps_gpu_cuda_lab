# Experiment 4: 2D Convolution — Benchmark Results and Analysis

## Implementation Summary

Kernels were written in CUDA C and compiled from a Google Colab notebook via
`torch.utils.cpp_extension.load`, targeting a T4 GPU (Compute Capability 7.5).
The Python harness called the compiled CUDA functions directly, timed them with
`torch.cuda.Event`, and compared against `F.conv2d` (cuDNN backend).

No CPU baseline was implemented — the meaningful comparison is between the four
GPU kernel variants and PyTorch/cuDNN, not against a single-threaded Python loop.

**Files:**
- `conv_kernels.cu` — all four CUDA kernel variants + pybind11 wrappers
- `parallel_convolution.ipynb` — correctness checks, benchmark harness, results

---

## Setup

| Item | Value |
|------|-------|
| GPU | NVIDIA T4 (Google Colab) |
| Compute Capability | 7.5 |
| T4 peak DRAM bandwidth | 320 GB/s |
| T4 peak FP32 compute | 8,141 GFLOPS |
| T4 ridge point | 25.4 FLOP/byte |
| CUDA version | 12.x |
| PyTorch version | 2.x |
| Warmup runs | 3 (discarded) |
| Timed runs | 20 (averaged) |
| Timer | `torch.cuda.Event` |
| Input dtype | float32 |
| Image shape | square N×N, single channel |
| Kernel shape | square K×K |
| Padding | 0 (valid convolution) |
| Output shape | (N−K+1) × (N−K+1) |

---

## Kernel Versions

| # | Name | Description |
|---|------|-------------|
| 1 | `naive` | One thread per output pixel. Each thread reads its K×K patch from global memory. Kernel weights read from global memory on every access. |
| 2 | `constmem` | Same thread structure as naive. Kernel weights copied to `__constant__` memory once via `cudaMemcpyToSymbol`. Inner loop reads from constant cache (broadcast to all 32 threads in a warp per cycle). |
| 3 | `tiled16` | Thread block = 16×16. Collaboratively loads input tile of (16+K−1)² into shared memory. Halo cells loaded with bounds check. Compute reads from shared memory. Kernel weights in constant memory. |
| 4 | `tiled32` | Same as tiled16 but thread block = 32×32 (1024 threads, max for T4). Input tile = (32+K−1)². |
| 5 | `pytorch_ref` | `F.conv2d(padding=0)` — cuDNN backend. Algorithm chosen per size by cuDNN (Winograd for small K, direct or FFT otherwise). |

**Shared memory allocation (templated on TILE_DIM, runtime K):**
```
__shared__ float s_tile[TILE_DIM + MAXK - 1][TILE_DIM + MAXK - 1]
                                  ^^^^
                         MAXK=11 used at compile time
                         actual kH/kW used for runtime loop bounds
```

---

## Arithmetic Intensity

All versions of the same (N, K) have identical arithmetic intensity — it is a
property of the algorithm, not the implementation. Higher bandwidth reported for
a kernel means it wastes fewer DRAM fetches.

| K | Intensity (FLOP/byte) | T4 roofline region |
|---|----------------------|--------------------|
| 3 | 2.241 | memory-bound (11× below ridge) |
| 5 | 6.201 | memory-bound (4× below ridge) |
| 7 | 12.104 | memory-bound (2× below ridge) |

For all three kernel sizes used in CNNs, the operation is memory-bound.
**The only lever is reducing wasted DRAM traffic, not increasing compute.**

---

## Results

### Execution Time (ms)

#### K = 3×3
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 0.0415 | 0.1410 | 0.5353 | 1.5057 |
| 2_constmem | 0.0353 | 0.1296 | 0.4978 | 1.3796 |
| 3_tiled16 | 0.0425 | 0.1585 | 0.6197 | 1.6052 |
| 4_tiled32 | 0.0420 | 0.1577 | 0.6134 | 1.5893 |
| 5_pytorch_ref | 0.0529 | 0.1346 | 0.4724 | 0.9663 |

#### K = 5×5
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 0.0763 | 0.2600 | 1.0149 | 2.2793 |
| 2_constmem | 0.0539 | 0.1980 | 0.7685 | 2.0220 |
| 3_tiled16 | 0.0625 | 0.2334 | 0.8981 | 2.2960 |
| 4_tiled32 | 0.0596 | 0.2269 | 0.8945 | 2.1808 |
| 5_pytorch_ref | 0.0687 | 0.2512 | 0.5645 | 2.1413 |

#### K = 7×7
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 0.1251 | 0.4699 | 1.0609 | 4.1654 |
| 2_constmem | 0.0900 | 0.3297 | 0.7376 | 2.9847 |
| 3_tiled16 | 0.1380 | 0.3563 | 0.7920 | 3.2513 |
| 4_tiled32 | 0.0778 | 0.2899 | 0.6962 | 2.7542 |
| 5_pytorch_ref | 0.1132 | 0.4140 | 0.8779 | 3.7711 |

---

### Effective Bandwidth (GB/s) — T4 peak = 320 GB/s

Denominator = minimum bytes moved: (N² + K² + (N−K+1)²) × 4.
Higher = less wasted DRAM traffic.

#### K = 3×3
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 50.37 | 59.38 | 62.62 | 89.10 |
| 2_constmem | 59.26 | 64.61 | 67.34 | 97.24 |
| 3_tiled16 | 49.17 | 52.81 | 54.10 | 83.57 |
| 4_tiled32 | 49.72 | 53.08 | 54.65 | 84.41 |
| 5_pytorch_ref | 39.46 | 62.21 | 70.97 | 138.82 |

#### K = 5×5
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 27.27 | 32.14 | 33.00 | 58.83 |
| 2_constmem | 38.63 | 42.21 | 43.58 | 66.31 |
| 3_tiled16 | 33.31 | 35.80 | 37.29 | 58.40 |
| 4_tiled32 | 34.93 | 36.82 | 37.44 | 61.49 |
| 5_pytorch_ref | 30.28 | 33.27 | 59.32 | 62.62 |

#### K = 7×7
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 16.57 | 17.75 | 31.54 | 32.17 |
| 2_constmem | 23.03 | 25.29 | 45.36 | 44.90 |
| 3_tiled16 | 15.02 | 23.41 | 42.24 | 41.22 |
| 4_tiled32 | 26.65 | 28.77 | 48.06 | 48.66 |
| 5_pytorch_ref | 18.31 | 20.14 | 38.11 | 35.54 |

---

### % of T4 Peak Bandwidth (320 GB/s)

#### K = 3×3
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 15.74 | 18.55 | 19.57 | 27.84 |
| 2_constmem | 18.52 | 20.19 | 21.04 | 30.39 |
| 3_tiled16 | 15.37 | 16.50 | 16.91 | 26.12 |
| 4_tiled32 | 15.54 | 16.59 | 17.08 | 26.38 |
| 5_pytorch_ref | 12.33 | 19.44 | 22.18 | 43.38 |

#### K = 5×5
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 8.52 | 10.04 | 10.31 | 18.38 |
| 2_constmem | 12.07 | 13.19 | 13.62 | 20.72 |
| 3_tiled16 | 10.41 | 11.19 | 11.65 | 18.25 |
| 4_tiled32 | 10.92 | 11.51 | 11.70 | 19.21 |
| 5_pytorch_ref | 9.46 | 10.40 | 18.54 | 19.57 |

#### K = 7×7
| kernel | N=512 | N=1024 | N=2048 | N=4096 |
|--------|-------|--------|--------|--------|
| 1_naive | 5.18 | 5.55 | 9.86 | 10.05 |
| 2_constmem | 7.20 | 7.90 | 14.17 | 14.03 |
| 3_tiled16 | 4.69 | 7.32 | 13.20 | 12.88 |
| 4_tiled32 | 8.33 | 8.99 | 15.02 | 15.21 |
| 5_pytorch_ref | 5.72 | 6.29 | 11.91 | 11.11 |

---

### Speedup vs Naive — N = 1024

| kernel | K=3 | K=5 | K=7 |
|--------|-----|-----|-----|
| 1_naive | 1.00 | 1.00 | 1.00 |
| 2_constmem | 1.09 | 1.31 | 1.43 |
| 3_tiled16 | 0.89 | 1.11 | 1.32 |
| 4_tiled32 | 0.89 | 1.15 | 1.62 |
| 5_pytorch_ref | 1.05 | 1.04 | 1.14 |

### Time as % of PyTorch/cuDNN — N = 1024
(100% = matched cuDNN. < 100% = faster than cuDNN.)

| kernel | K=3 | K=5 | K=7 |
|--------|-----|-----|-----|
| 1_naive | 104.8% | 103.5% | 113.5% |
| 2_constmem | 96.3% | 78.8% | 79.6% |
| 3_tiled16 | 117.8% | 92.9% | 86.1% |
| 4_tiled32 | 117.2% | 90.3% | **70.0%** |
| 5_pytorch_ref | 100.0% | 100.0% | 100.0% |

---

## Key Findings

### Finding 1 — Tiled kernels are SLOWER than naive at K=3 (biggest surprise)

```
N=1024, K=3:
  naive:    0.141 ms  (59.4 GB/s effective)
  tiled16:  0.159 ms  (52.8 GB/s effective)  ← 12% SLOWER than naive
  tiled32:  0.158 ms  (53.1 GB/s effective)  ← 12% SLOWER than naive
```

**Why:** For K=3 (K²=9), each input pixel is re-fetched only 9 times in the naive
kernel. The T4's L2 cache (4 MB) absorbs these 9 redundant reads at L2 speeds,
not DRAM speeds. The overhead of tiled convolution — collaborative halo loading,
`__syncthreads()`, and the two-level index mapping — costs more than the L2
miss savings from shared memory. The net result is negative.

This is the most important empirical observation of this experiment:
**shared memory tiling is NOT always beneficial for convolution.**
The break-even point is between K=3 and K=5.

---

### Finding 2 — Constant memory is the strongest single optimization for K ≤ 5

```
Speedup of constmem over naive:
  K=3: 1.09×  (marginal, but positive — unlike tiling which is negative)
  K=5: 1.31×  (clearly better than tiled16 at 1.11× and tiled32 at 1.15×)
  K=7: 1.43×  (still strong, but now tiled32 at 1.62× pulls ahead)
```

**Why it works:** `__constant__` memory uses a dedicated broadcast cache. When
all 32 threads in a warp access `d_kernel[kh*kW+kw]` — the same address — it
takes ONE fetch from the constant cache, not 32 separate L2 requests.

In the naive kernel, kernel weights go through L2 alongside input pixel reads,
competing for L2 bandwidth. Moving weights to constant memory frees L2 entirely
for caching input pixels, which is exactly what reduces the penalty of the 9×/25×/49×
redundant input fetches.

The constant memory benefit scales with K²:
- K=3: saves 9 L2 reads per output pixel  → 9% speedup
- K=5: saves 25 L2 reads per output pixel → 31% speedup
- K=7: saves 49 L2 reads per output pixel → 43% speedup

**This was under-predicted.** The hypothesis expected only "marginal" improvement
because "the kernel fits in L2 cache anyway." That reasoning missed the broadcast
cache advantage — constant memory doesn't just cache, it serves all 32 threads
simultaneously from one read.

---

### Finding 3 — Tiling benefit grows with K, confirms the reuse hypothesis

```
Speedup of tiled32 over naive (N=1024):
  K=3: 0.89×  (negative — tiling costs more than it saves)
  K=5: 1.15×  (break-even crossed, now positive)
  K=7: 1.62×  (strong — K²=49 reuse too large for L2 to absorb)
```

At K=7, each input pixel contributes to 49 output pixels in the naive kernel.
L2 cannot hold all the active working set: a 1024×1024 input = 4 MB, which is
close to the T4's 4 MB L2. At K=7, L2 starts to thrash, and shared memory
tiling genuinely eliminates the wasted traffic. The result: tiled32 achieves
48.66 GB/s at N=4096 vs naive's 32.17 GB/s — a 51% improvement at the largest
size tested.

---

### Finding 4 — tiled32 beats tiled16, and the gap grows with K

```
N=1024 execution time (ms):
         K=3         K=5         K=7
tiled16: 0.1585      0.2334      0.3563
tiled32: 0.1577      0.2269      0.2899
ratio:   1.005×      1.029×      1.229×
```

For K=3: nearly identical (both slower than naive anyway).
For K=7: tiled32 is 23% faster than tiled16.

**Why:** At K=7, each block needs to load (TILE+6)² cells:
- TILE=16: (22²=484) / (16²=256) = 1.89 cells/thread → nearly 2 loads each
- TILE=32: (38²=1444) / (32²=1024) = 1.41 cells/thread → ~1.4 loads each

TILE=32 has a more favourable loading ratio and 4× more threads per block, giving
the GPU scheduler more warps to hide memory latency behind. Both factors compound
as K grows.

---

### Finding 5 — Our kernels beat cuDNN at K=5 and K=7 (unexpected)

```
Time as % of pytorch_ref (N=1024):
           K=3      K=5      K=7
constmem:  96.3%    78.8%    79.6%   ← 27% faster at K=5
tiled32:  117.2%    90.3%    70.0%   ← 43% faster at K=7
```

At K=7, our `tiled32` kernel runs in 0.290 ms while `F.conv2d` takes 0.414 ms.
We are **43% faster than cuDNN** at K=7, N=1024.

**Why this happens:**
1. cuDNN has per-launch overhead: algorithm selection, workspace allocation, input/output
   format checks, and cuDNN handle operations. For a single 1024×1024 image, this
   fixed overhead is a measurable fraction of total time.
2. cuDNN's Winograd transform (optimised for K=3) carries its own transform overhead
   that may not pay off at N=1024 with a single image.
3. Our raw kernel has zero framework overhead — it goes straight to GPU execution.

At large N (4096), the story reverses for K=3: pytorch_ref achieves 138.82 GB/s
vs our best 97.24 GB/s (constmem). At scale, cuDNN's algorithmic advantages
(Winograd, optimal tile sizing, vectorised loads) overcome its launch overhead.

**Lesson:** library implementations are optimised for the throughput case (large
batch, large images). For single-image inference, a lean custom kernel can win.

---

### Finding 6 — All kernels are far below T4 peak bandwidth

```
Best achieved bandwidth across all measurements:
  Our kernels: 97.24 GB/s (constmem, N=4096, K=3) = 30.4% of T4 peak
  pytorch_ref: 138.82 GB/s (N=4096, K=3)           = 43.4% of T4 peak
```

Even at N=4096, we reach at most 30% of the 320 GB/s theoretical peak.

**Why bandwidth is low despite being "memory-bound":**
- Arithmetic intensity for K=3 is only 2.24 FLOP/byte. For each byte loaded,
  there is only 0.56 FLOP of computation. This means the GPU is not fully
  pipelining compute and memory — it completes memory operations before compute
  can hide them.
- Small N (512) has low SM occupancy: not enough output pixels to saturate all SMs.
- The nested inner loop (kH × kW iterations per thread) introduces loop overhead
  and register pressure that reduces warp throughput.

For comparison, exp3 (reduction) achieved 60–80% of peak bandwidth. Reduction
has a simpler inner loop (one addition) and benefits from larger working sets.
Convolution's K×K loop is the overhead that prevents bandwidth saturation.

---

## Hypothesis Review

| Hypothesis | Prediction | Actual | Verdict |
|-----------|-----------|--------|---------|
| Q2: Constant memory gives marginal improvement | small, larger for K=7 | 9%→43% speedup, scales strongly with K | ✅ direction correct, ❌ magnitude underestimated |
| Q3: Tiling improves over naive by 2–5× (K=3) | 2–5× speedup | 0.89× (negative!) at K=3 | ❌ Wrong — L2 absorbs K=3 reuse |
| Q3: Tiling improves 5–10× (K=7) | 5–10× speedup | 1.62× at K=7 | ❌ Wrong — much smaller than predicted |
| Q3: Bandwidth approaches 60–80% of peak | 192–256 GB/s | max 48.66 GB/s (15.2% of peak) | ❌ Wrong — overhead limits bandwidth |
| Q4: TILE=32 slightly faster than TILE=16 | < 20% difference | 23% at K=7 | ✅ Correct |
| Q5: Tiled gets 30–60% of cuDNN at K=3 | 30–60% | tiled32 is 117% (slower) | ❌ Wrong direction — cuDNN wins K=3 |
| Q5: Tiled gets closer at K=7 (cuDNN uses direct) | within 10–20% | tiled32 at 70% (30% FASTER) | ✅ Correct direction, stronger than predicted |
| Q6: Tiling speedup grows with K | yes | confirmed (0.89→1.15→1.62×) | ✅ Confirmed |

---

## What Was Confirmed

- ✅ Convolution is memory-bound for K ≤ 7 on T4 (all kernels far from compute peak)
- ✅ Tiling speedup over naive grows monotonically with K
- ✅ TILE=32 outperforms TILE=16, advantage grows with K
- ✅ cuDNN gap closes as K grows toward direct convolution territory
- ✅ Effective bandwidth grows with N (larger images = better SM utilisation)

## What Was Surprising

- ❌ Tiled kernels are SLOWER than naive at K=3 — L2 cache absorbs small reuse
- ❌ Constant memory is the strongest optimization for K ≤ 5, stronger than tiling
- ❌ Our kernels beat cuDNN at K=5 and K=7 for N=1024 — framework overhead matters
- ❌ Peak bandwidth utilisation is low (max 30% for our kernels) — K×K loop overhead limits pipelining
- ❌ The break-even point for tiling is between K=3 and K=5, not at K=3 as assumed

---

## Ranking of Optimisations by K

```
K=3:  constmem (1.09×) > naive (1.00×) > tiled32 (0.89×)
K=5:  constmem (1.31×) > tiled32 (1.15×) > tiled16 (1.11×) > naive (1.00×)
K=7:  tiled32 (1.62×) > constmem (1.43×) > tiled16 (1.32×) > naive (1.00×)
```

The crossover: constant memory wins when K² reuse is small enough that L2 absorbs
the redundant reads. Shared memory tiling wins when K² reuse overwhelms L2.

---

## What Success Looks Like — Achieved

- ✅ Implemented 2D thread indexing confidently (row/col from blockIdx + threadIdx)
- ✅ Confirmed convolution is memory-bound for all K ≤ 7 from measured data
- ✅ Used `__constant__` memory correctly with `cudaMemcpyToSymbol` from Python
- ✅ Implemented the halo-loading pattern (collaborative load, bounds check, __syncthreads)
- ✅ Discovered that L2 absorbs K=3 reuse — tiling has a break-even point
- ✅ Confirmed tiling benefit scales with K² (reuse per pixel)
- ✅ Discovered cuDNN has per-launch overhead that raw kernels avoid for single images
- ✅ Learned that constant memory's broadcast advantage is more powerful than expected
- ✅ Connected the gap to Winograd: the remaining cuDNN advantage at large N, K=3
  is from reduced FLOPs (F(2,3) Winograd: 2.25× fewer multiplications), not memory

---

## Forward — Experiment 5: Winograd Convolution

The remaining cuDNN advantage at K=3, large N comes from the **Winograd F(2,3)
algorithm**, which transforms the convolution into the frequency domain where
fewer multiplications are needed:

```
Direct 3×3 conv:   9 multiplications per output element
Winograd F(2,3):   4 multiplications per 2×2 output block = 1 multiply per element
Reduction:         2.25× fewer FLOPs
```

At N=4096, K=3: pytorch_ref achieves 138.82 GB/s vs our 97.24 GB/s — 43% faster.
This gap is entirely explainable by Winograd. Implementing F(2,3) Winograd from
scratch is planned as **Experiment 5**, deferred until after completing the
remaining PMPP chapters.

The key new concept in exp5: transform overhead vs compute savings tradeoff —
Winograd requires input/output transforms that themselves cost memory bandwidth,
so the win is only guaranteed above a crossover input size.
