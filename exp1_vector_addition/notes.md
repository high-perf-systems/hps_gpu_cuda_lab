# Notes: Experiment 1 — Vector Addition

## Hardware & Software Setup

| Item | Value |
|------|-------|
| **GPU** | NVIDIA Tesla T4 |
| **Architecture** | Turing (Compute Capability 7.5) |
| **Streaming Multiprocessors** | 40 SMs |
| **Global Memory** | 15,637 MB |
| **Shared Memory per Block** | 48 KB |
| **Peak Memory Bandwidth** | 320 GB/s |
| **Driver Version** | 580.82.07 |
| **CUDA Version** | 12.8 (nvcc) / 13.0 (driver) |
| **OS** | Linux (Google Colab) |
| **Build Flags** | `nvcc -O2` |
| **Threads per Block** | 256 |
| **Warmup Runs** | 3 (discarded) |
| **Timed Runs** | 10 (averaged) |

---

## Benchmark Results

### Table 1: Kernel Performance (compute only, excludes transfers)

| N | CPU (ms) | GPU Pageable (ms) | GPU Pinned (ms) | Speedup Pageable | Speedup Pinned |
|---|----------|-------------------|-----------------|-----------------|----------------|
| 1,024 (1<<10) | 0.0009 | 0.0082 | 0.0078 | 0.1x | 0.1x |
| 65,536 (1<<16) | 0.0623 | 0.0138 | 0.0100 | 4.5x | 6.2x |
| 1,048,576 (1<<20) | 0.6959 | 0.0535 | 0.0554 | 13.0x | 12.6x |
| 4,194,304 (1<<22) | 3.4055 | 0.2016 | 0.1938 | 16.9x | 17.6x |
| 16,777,216 (1<<24) | 17.6178 | 0.7770 | 0.7796 | 22.7x | 22.6x |

---

### Table 2: Transfer Performance (pinned vs pageable)

| N | H2D Pageable (ms) | H2D Pinned (ms) | H2D Speedup | D2H Pageable (ms) | D2H Pinned (ms) | D2H Speedup |
|---|-------------------|-----------------|-------------|-------------------|-----------------|-------------|
| 1,024 | 0.0223 | 0.0248 | 0.9x | 0.0138 | 0.0102 | 1.4x |
| 65,536 | 0.1752 | 0.0692 | 2.5x | 0.0927 | 0.0340 | 2.7x |
| 1,048,576 | 1.8105 | 0.7532 | 2.4x | 0.9666 | 0.3318 | 2.9x |
| 4,194,304 | 6.9156 | 2.7358 | 2.5x | 3.6646 | 1.2901 | 2.8x |
| 16,777,216 | 32.7281 | 10.9999 | 3.0x | 16.7069 | 5.1339 | 3.3x |

---

### Table 3: Memory Bandwidth (kernel only)

| N | Pageable (GB/s) | Pinned (GB/s) | % of T4 Peak (320 GB/s) |
|---|-----------------|---------------|--------------------------|
| 1,024 | 1.5 | 1.6 | 0.5% |
| 65,536 | 57.1 | 78.5 | 24.5% |
| 1,048,576 | 235.1 | 227.1 | 70.9% |
| 4,194,304 | 249.6 | 259.7 | 81.1% |
| 16,777,216 | 259.1 | 258.2 | 80.7% |

---

### Table 4: Total Time (kernel + H2D + D2H)

| N | CPU (ms) | Total Pageable (ms) | Total Pinned (ms) | Speedup Pageable | Speedup Pinned |
|---|----------|---------------------|-------------------|-----------------|----------------|
| 1,024 | 0.0009 | 0.0443 | 0.0427 | 0.0x | 0.0x |
| 65,536 | 0.0623 | 0.2817 | 0.1133 | 0.2x | 0.6x |
| 1,048,576 | 0.6959 | 2.8306 | 1.1403 | 0.2x | 0.6x |
| 4,194,304 | 3.4055 | 10.7818 | 4.2197 | 0.3x | 0.8x |
| 16,777,216 | 17.6178 | 50.2120 | 16.9134 | 0.4x | 1.0x |

---

### Table 5: Crossover Analysis

| Crossover Type | N Value |
|---------------|---------|
| Kernel crossover (GPU kernel < CPU) | N >= 65,536 (1<<16) |
| Total crossover (GPU total < CPU) | N >= 16,777,216 (1<<24) |
| Gap between crossovers | 256x difference in N |

---

## nvprof Timeline Analysis

### Profiling Note

nvprof **metric and event collection is not supported** for
Compute Capability 7.5+ (Tesla T4 = Turing architecture).
NVIDIA's official replacement tools are:
- **Nsight Compute (ncu)**: kernel-level performance metrics
- **Nsight Systems (nsys)**: system-wide timeline profiling

Basic timeline profiling (`nvprof` without `--metrics`) still works
and provides GPU activity breakdown, call counts, and API overhead.

---

### nvprof Timeline: N = 4,194,304 (1<<22)

```
==NVPROF== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:
   63.60%  126.25ms   52  2.4278ms  1.3567ms  4.0863ms  [CUDA memcpy HtoD]
   33.88%   67.26ms   26  2.5869ms  1.2741ms  10.387ms  [CUDA memcpy DtoH]
    2.51%    4.99ms   26   191.9us   188.9us   193.1us   vector_add_kernel

 API calls:
   47.41%  205.44ms   78  2.6338ms          cudaMemcpy
   45.11%  195.49ms    3  65.16ms           cudaHostAlloc
    1.42%    6.14ms    1              6.14ms  cudaGetDeviceProperties
    1.12%    4.84ms   60    80.6us           cudaEventSynchronize
    0.29%    1.26ms   26    27.9us           cudaLaunchKernel
```

#### Call Count Verification

| Operation | Count | Expected | Match? |
|-----------|-------|----------|--------|
| HtoD memcpy | 52 | 2 arrays × (3 warmup + 10 timed) × 2 versions = 52 | ✅ |
| DtoH memcpy | 26 | 1 array × 13 runs × 2 versions = 26 | ✅ |
| Kernel launches | 26 | 1 kernel × 13 runs × 2 versions = 26 | ✅ |

Call counts exactly match our benchmark structure —
confirms the warmup/timed loop is executing correctly.

#### GPU Activity Breakdown

| Activity | Time % | Interpretation |
|----------|--------|----------------|
| HtoD transfers | 63.60% | Copying A + B to GPU |
| DtoH transfers | 33.88% | Copying C back to CPU |
| Kernel execution | 2.51% | Actual computation |
| **Transfers total** | **97.49%** | **Data movement dominates!** |

The kernel does the work in 2.51% of GPU time.
The remaining 97.49% is moving data between CPU and GPU.
This is the defining characteristic of memory-bound kernels.

#### Kernel Consistency (Min/Max spread)

```
vector_add_kernel: min=188.86us, max=193.08us → spread = 2.2%
```

Very tight spread — warmup was effective, measurements are stable.

#### API Overhead Findings

```
cudaHostAlloc: 3 calls, avg 65.16ms each = 195.49ms total
```

Pinned memory allocation is **extremely expensive** as a one-time cost:
- Must lock physical RAM pages (prevent OS swapping)
- Must register with CUDA driver
- Must map into GPU virtual address space
- For 16.78 MB × 3 arrays, each allocation touches ~1,000 pages

This confirms why production code **allocates once and reuses**
rather than allocating per-frame or per-invocation.

```
cudaLaunchKernel: 26 calls, avg 27.9us per launch
```

Kernel launch overhead = 27.9μs CPU-side
vs kernel runtime = 191.9μs GPU-side
Launch overhead = **14.5% of kernel time** at N=4M.

For shorter kernels (small N), this ratio gets worse rapidly,
contributing to GPU underperformance at small problem sizes.

---

### nvprof Timeline: N = 65,536 (1<<16) — For Contrast

```
==NVPROF== Profiling result:
            Type  Time(%)      Time     Calls      Avg
 GPU activities:
   64.60%  1.2431ms   52   23.9us  [CUDA memcpy HtoD]
   29.15%  0.5610ms   26   21.6us  [CUDA memcpy DtoH]
    6.24%  0.1202ms   26    4.6us  vector_add_kernel
```

At N=64K, kernel is 6.24% of GPU time vs 2.51% at N=4M.
Kernel is still dominated by transfers, but the ratio improves
slightly at smaller N due to lower absolute transfer volume.

However, kernel time is only 4.6μs — compared to
cudaLaunchKernel overhead of ~18.8μs (from API calls).
**Launch overhead is 4x larger than the kernel itself at N=64K!**

---

## Key Observations

### Observation 1: Two Distinct Crossover Points

The experiment reveals a critical distinction between
**kernel-only** and **total** performance crossover:

```
Kernel crossover:  N >= 65,536   (GPU kernel < CPU)
Total crossover:   N >= 16,777,216 (GPU total < CPU)
Gap:               256x difference in N
```

At N=65,536: GPU kernel is 6.2x faster than CPU (kernel only).
But total time including transfers: GPU is 2.5x **slower**!

This gap exists because for a single kernel invocation,
transfer cost dominates.
The GPU's kernel advantage only overcomes transfer overhead
at very large N (16M+ elements).

**Practical implication**: GPU wins in real applications when:
- Data already lives on GPU from a previous operation
- Many kernels share the same uploaded data
- Transfers are amortized across multiple operations

---

### Observation 2: Bandwidth Saturation Pattern

```
N =     1,024:   1.6 GB/s  →   0.5% of peak  (36/40 SMs idle!)
N =    65,536:  78.5 GB/s  →  24.5% of peak
N = 1,048,576: 253.5 GB/s  →  79.2% of peak  ← large jump here
N = 4,194,304: 259.7 GB/s  →  81.1% of peak  ← nearly flat
N =16,777,216: 258.2 GB/s  →  80.7% of peak  ← saturated
```

The critical transition happens **between N=64K and N=1M**:

- N=64K: 256 blocks. Not enough to fill all 40 SMs fully.
  SMs waiting for work, memory pipeline underloaded.

- N=1M: 4,096 blocks. All 40 SMs active and busy.
  Memory pipeline fully loaded, bandwidth saturated.

Beyond N=1M, adding more elements gives proportionally more
time — no further efficiency gain possible. The memory
controller is the bottleneck.

**Why ~81% and not 100%?**

The ~19% gap from theoretical peak (320 GB/s) is expected:

| Source of Loss | Estimated Impact |
|---------------|-----------------|
| ECC memory protection (enabled by default on T4) | ~6% |
| Address calculation per warp | ~3% |
| DRAM row open/close overhead | ~4% |
| Boundary guard (if idx >= N) check | ~1% |
| Memory request scheduling | ~5% |

~81% efficiency is normal for production CUDA code.
Most real GPU applications achieve 70-85% of peak bandwidth.

---

### Observation 3: Transfers Dominate Execution Time

At N=4M with pinned memory:

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| H2D transfer (A+B) | 2.74 | 64.9% |
| D2H transfer (C) | 1.29 | 30.6% |
| Kernel execution | 0.19 | 4.5% |
| **Total** | **4.22** | **100%** |

Only 4.5% of time is actual computation. 95.5% is data movement.

This is consistent across N values once N is large enough:
nvprof confirms 97.49% GPU time in transfers at N=4M.

**This is the fundamental challenge for GPU computing**:
getting data to/from the device efficiently matters far more
than optimizing the kernel itself for memory-bound problems.

---

### Observation 4: Pinned Memory 2.5-3.3x Faster for Transfers

Transfer speedup (pinned vs pageable) scales with N:

```
N =     1,024:  H2D 0.9x  (no benefit — noise level, DMA overhead > data)
N =    65,536:  H2D 2.5x  (DMA benefit starts to show)
N = 4,194,304:  H2D 2.5x  (consistent benefit)
N =16,777,216:  H2D 3.0x  (largest benefit at large N)
```

Why pageable is slower:

```
Pageable path (2 copies):
    CPU RAM (pageable) → CPU RAM (internal pinned staging) → GPU VRAM

Pinned path (1 copy):
    CPU RAM (pinned) → GPU VRAM (DMA direct, no staging needed)
```

The extra CPU→CPU copy for pageable memory costs proportionally
more at larger N, explaining why the speedup grows with N.

**At N=1K, pinned is actually slightly SLOWER** (0.9x for H2D).
At this tiny size, DMA setup overhead exceeds the copy savings.
Pinned memory is only beneficial beyond approximately N=16K.

**Kernel time is identical for both** (pageable vs pinned):
at N=4M, the difference is 0.2016ms vs 0.1938ms = 4% — noise level.
The kernel executes on GPU memory and has no knowledge of how
host memory was allocated. Only the transfer path differs.

---

### Observation 5: D2H is ~Half the Time of H2D

Consistent pattern across all N values:

```
N=16M: H2D=32.73ms, D2H=16.71ms → D2H is 0.51x of H2D
N= 4M: H2D= 6.92ms, D2H= 3.66ms → D2H is 0.53x of H2D
N= 1M: H2D= 1.81ms, D2H= 0.97ms → D2H is 0.53x of H2D
```

This is **not** a hardware asymmetry in the PCIe bus.

We copy **two arrays to GPU** (A and B: 2 × N × 4 bytes)
but only **one array back** (C: 1 × N × 4 bytes).

Total bytes H2D = 2 × N × sizeof(float)
Total bytes D2H = 1 × N × sizeof(float)
Expected ratio  = 2:1 → observed ~1.96:1 ✅

Per-array bandwidth is symmetric in both directions.

---

### Observation 6: Small N — GPU Dramatically Slower

At N=1,024:

```
CPU kernel:        0.0009 ms  ←  fastest
GPU kernel (pin):  0.0078 ms  ←  8.7x slower than CPU
GPU total  (pin):  0.0427 ms  ←  47x slower than CPU
```

Three reasons GPU loses at small N:

**1. SM underutilization**
N=1024 → only 4 blocks launched across 40 SMs.
36 SMs have no work at all. GPU hired 40 workers,
assigned work only to 4 of them.

**2. Bandwidth starvation**
Only 1.6 GB/s achieved vs 320 GB/s peak (0.5% utilization).
Memory controller barely tickled — most latency is overhead,
not actual data transfer time.

**3. Launch overhead dominates**
Kernel launch: ~28μs (API call overhead)
Kernel runtime: 7.8μs
Launch overhead is **3.6x larger** than the actual work!

This is why GPU programming always requires "sufficiently large"
problems. The break-even point for kernel-only is N~65K for
this particular kernel on Tesla T4.

---

### Observation 7: Run-to-Run Variance and Warmup Effectiveness

From nvprof Min/Max at N=4M:

```
vector_add_kernel:  min=188.86us  max=193.08us  spread=2.2%
HtoD memcpy:        min=1.356ms   max=4.086ms   spread=201%
DtoH memcpy:        min=1.274ms   max=10.387ms  spread=716%
```

Key insight: **kernel time is very stable** (2.2% spread),
but **transfer times are highly variable** (up to 716% spread!).

Kernel stability confirms warmup was effective for GPU compute.
Transfer variability is due to:
- PCIe bus contention with other system activity
- OS memory management (page faults, TLB misses)
- DMA engine scheduling non-determinism

This is why averaging 10 runs matters — a single transfer
measurement can be 7x off from the true average.
This mirrors our CPU lab finding: cold cache = high variance,
warm cache = stable measurements.

---

## Hypothesis Validation

| Hypothesis | Prediction | Actual | Verdict |
|-----------|-----------|--------|---------|
| GPU kernel faster for large N | Yes | Yes — 22.7x at N=16M | ✅ Confirmed |
| GPU slower for small N | Yes | 8.7x slower at N=1K (kernel) | ✅ Confirmed |
| Transfers dominate execution | Yes | 95.5% of total time at N=4M | ✅ Strongly confirmed |
| Pinned memory faster (20-40%) | 20-40% faster | 2.5-3.3x faster (150-230%) | ⚠️ Direction correct, magnitude underestimated |
| Peak bandwidth ~190-250 GB/s | 190-250 GB/s | ~260 GB/s | ✅ Close, slightly conservative |
| GPU warmup effect visible | Yes | 430x slower on cold first run (earlier experiment) | ✅ Confirmed |
| Pinned kernel time = pageable | No difference | <5% difference (noise) | ✅ Confirmed |
| D2H ~half of H2D time | ~0.5x | 0.51-0.53x | ✅ Confirmed exactly |

**Biggest surprise**: Pinned memory transfer benefit was far larger
than predicted (2.5-3.3x vs predicted 20-40%). The extra
CPU→CPU staging copy for pageable memory carries more overhead
than intuition suggests, especially at large N.

---

## nvprof Metrics Limitation

Hardware performance counters (occupancy, gld_efficiency,
dram_throughput, etc.) are **not collectible** via nvprof on
Compute Capability 7.5+. Theoretical values based on
access pattern analysis:

| Metric | Expected Value | Reasoning |
|--------|---------------|-----------|
| achieved_occupancy | ~85-90% | Few registers, no shared memory |
| gld_efficiency | ~99-100% | idx = blockIdx.x * blockDim.x + threadIdx.x → perfect sequential access → perfect coalescing |
| gst_efficiency | ~99-100% | Same sequential access pattern for writes |
| L2 read hit rate | ~0-5% | Each element read exactly once, no reuse possible |
| dram_read_throughput | ≈ gld_throughput | No L2 reuse → all reads go to DRAM |

The L2 hit rate being near zero is important:
for vector addition, every element is accessed exactly once.
There is no data reuse to exploit with caching.
This changes dramatically in Experiment 2 (matrix multiply):
with shared memory tiling, the same data is read many times,
L2/shared memory hit rate becomes high,
and effective bandwidth per DRAM byte accessed increases.

**ncu (Nsight Compute)** is the correct tool for CC 7.5+.
Will attempt ncu for Experiment 2 if available in environment.
