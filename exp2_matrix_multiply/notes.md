# Notes: Experiment 2 — Matrix Multiplication

## Hardware & Software Setup

| Item | Value |
|------|-------|
| **GPU** | NVIDIA Tesla T4 |
| **Architecture** | Turing (Compute Capability 7.5) |
| **Streaming Multiprocessors** | 40 SMs |
| **Global Memory** | 15,637 MB |
| **Shared Memory per Block** | 48 KB |
| **Peak Memory Bandwidth** | 320 GB/s |
| **Peak FP32 Compute** | 8,141 GFLOPS |
| **Driver Version** | 580.82.07 |
| **CUDA Version** | 12.8 (nvcc) / 13.0 (driver) |
| **OS** | Linux (Google Colab) |
| **Build Flags** | `nvcc -O2` |
| **Warmup Runs** | 3 (discarded) |
| **Timed Runs** | 10 (averaged) |

### Kernel Configurations

| Version | Block Dim | Threads/Block | Shared Mem/Block |
|---------|-----------|---------------|-----------------|
| CPU Baseline | N/A | N/A | N/A |
| GPU Naive | 16×16 | 256 | None |
| GPU Tiled T=16 | 16×16 | 256 | 2 × 16×16×4 = 2 KB |
| GPU Tiled T=32 | 32×32 | 1024 (T4 max!) | 2 × 32×32×4 = 8 KB |

---

## Benchmark Results

### Table 1: Kernel Time (ms)

| N | CPU (ms) | Naive GPU (ms) | Tiled T=16 (ms) | Tiled T=32 (ms) |
|---|----------|----------------|-----------------|-----------------|
| 256 | 22.143 | 0.1601 | 0.1110 | 0.1093 |
| 512 | 281.809 | 1.1600 | 0.7535 | 0.7114 |
| 1,024 | 3,239.836 | 9.1994 | 5.7991 | 3.0957 |
| 2,048 | ~81,211 (1 run) | 41.1958 | 26.0131 | 18.9269 |

---

### Table 2: Performance (GFLOPS) — T4 Peak FP32: 8,141 GFLOPS

| N | CPU | Naive GPU | Tiled T=16 | Tiled T=32 | % of Peak (T32) |
|---|-----|-----------|------------|------------|-----------------|
| 256 | 1.52 | 209.6 | 302.3 | 306.9 | 3.8% |
| 512 | 0.95 | 231.4 | 356.2 | 377.4 | 4.6% |
| 1,024 | 0.66 | 233.4 | 370.3 | 693.7 | 8.5% |
| 2,048 | — | 417.0 | 660.4 | 907.7 | 11.1% |

---

### Table 3: Speedup Over CPU Baseline

| N | Naive GPU | Tiled T=16 | Tiled T=32 |
|---|-----------|------------|------------|
| 256 | 138.3x | 199.5x | 202.5x |
| 512 | 242.9x | 374.0x | 396.2x |
| 1,024 | 352.2x | 558.7x | 1,046.6x |

---

### Table 4: Tiled Speedup Over Naive GPU

| N | T=16 vs Naive | T=32 vs Naive |
|---|---------------|---------------|
| 256 | 1.4x | 1.5x |
| 512 | 1.5x | 1.6x |
| 1,024 | 1.6x | 3.0x |
| 2,048 | 1.6x | 2.2x |

---

### Table 5: Effective Bandwidth (GB/s) — T4 Peak: 320 GB/s

| N | CPU | Naive GPU | Tiled T=16 | Tiled T=32 |
|---|-----|-----------|------------|------------|
| 256 | 0.0 | 4.9 | 7.1 | 7.2 |
| 512 | 0.0 | 2.7 | 4.2 | 4.4 |
| 1,024 | 0.0 | 1.4 | 2.2 | 4.1 |
| 2,048 | — | 1.2 | 1.9 | 2.7 |

Note: Bandwidth = 3×N²×4 / time (arithmetic minimum — read A, read B, write C).
Naive GPU actually reads MORE than this due to strided B access wasting cache lines.
Tiled GPU reads LESS per DRAM fetch due to shared memory reuse.
Low bandwidth for tiled does NOT mean slow — it means fewer DRAM accesses,
more compute per byte — exactly the design intent.

---

## nvprof Timeline Analysis (N=1024)

### Profiling Note
nvprof metric/event collection not supported for CC 7.5+ (Tesla T4).
Basic timeline profiling still works and provides GPU activity breakdown.

### GPU Activity Breakdown

```
Type       Time%     Time       Calls   Avg/Call   Min        Max
─────────────────────────────────────────────────────────────────────
Naive       46.20%  100.38ms    13      7.72ms     6.92ms     9.20ms
Tiled T=16  26.08%   56.67ms    13      4.36ms     4.36ms     4.36ms
Tiled T=32  24.27%   52.74ms    13      4.06ms     4.05ms     4.06ms
HtoD memcpy  2.10%    4.57ms     6     761us
DtoH memcpy  1.34%    2.91ms     3     970us
```

### Call Count Verification

| Operation | Count | Expected | Match? |
|-----------|-------|----------|--------|
| Kernel calls (each version) | 13 | 3 warmup + 10 timed = 13 | ✅ |
| HtoD memcpy | 6 | 2 matrices (A,B) × 3 versions = 6 | ✅ |
| DtoH memcpy | 3 | 1 matrix (C) × 3 versions = 3 | ✅ |

### Key nvprof Insight: Transfer vs Compute Ratio

```
Transfers (HtoD + DtoH): 2.10% + 1.34% = 3.44% of GPU time
Kernels:                 46.20% + 26.08% + 24.27% = 96.55% of GPU time
```

Complete REVERSAL from Experiment 1!

```
Experiment 1 (vector addition):
    Transfers: 97.5% of GPU time
    Kernel:     2.5% of GPU time

Experiment 2 (matrix multiply):
    Transfers:  3.4% of GPU time
    Kernels:   96.6% of GPU time
```

This is the direct consequence of higher arithmetic intensity.
Matrix multiply is COMPUTE BOUND — the GPU spends almost
all its time doing arithmetic, not moving data.

### Kernel Consistency (Min/Max spread at N=1024)

| Kernel | Min | Max | Spread |
|--------|-----|-----|--------|
| Naive | 6.92ms | 9.20ms | 32.9% |
| Tiled T=16 | 4.356ms | 4.362ms | 0.1% |
| Tiled T=32 | 4.052ms | 4.063ms | 0.3% |

Critical observation: naive kernel has HIGH variance (32.9%)
while tiled kernels are EXTREMELY consistent (<0.3% spread).

Why? Naive kernel's strided B access causes L2 cache miss storms —
memory access time varies with contention and DRAM scheduling.
Tiled kernels access shared memory predictably after the initial
coalesced DRAM load — execution time is deterministic.
Consistent performance = predictable real-time behavior.
This matters enormously for automotive perception pipelines!

### API Overhead

```
cudaMalloc: 9 calls, avg 20.1ms each = 181ms total
```

cudaMalloc is extremely expensive (physical page allocation + GPU mapping).
This confirms: NEVER allocate GPU memory per-frame in production!
Pre-allocate once, reuse every frame — same lesson as Experiment 1.

---

## Key Observations

### Observation 1: Complete Reversal of Bottleneck vs Experiment 1

The single most important finding of this experiment:

```
Vector addition (Exp 1):    Transfers dominate  (97.5% of time)
Matrix multiply (Exp 2):    Compute dominates   (96.6% of time)

Root cause: Arithmetic Intensity

Vector add:       1 FLOP per 12 bytes = 0.08 FLOP/byte
                  → memory bound, GPU starved for data

Naive matmul:     2N³ FLOPs / 12N² bytes = N/6 FLOP/byte
                  At N=1024: ~170 FLOP/byte → compute bound

Tiled T=32:       ~T× higher intensity than naive
                  → even more compute bound
```

T4 Ridge Point = Peak GFLOPS / Peak BW = 8141 / 320 = 25.4 FLOP/byte

At N=1024:
- Naive intensity (~170 FLOP/byte) >> ridge point (25.4) → compute bound ✅
- Tiled intensity (much higher) → even more compute bound ✅

The GPU is doing what it was designed for: massive parallel arithmetic!

---

### Observation 2: Tiled vs Naive — Speedup Grows with N

```
N=256:   T=32 vs Naive = 1.5x
N=512:   T=32 vs Naive = 1.6x
N=1024:  T=32 vs Naive = 3.0x  ← large jump!
N=2048:  T=32 vs Naive = 2.2x
```

Why does speedup grow with N?

At small N (256):
    Only 256/32 × 256/32 = 64 blocks launched
    Not enough blocks to fully saturate 40 SMs
    Both naive and tiled are underoccupied
    → Difference between them is small

At large N (1024, 2048):
    1024/32 × 1024/32 = 1024 blocks for T=32
    All 40 SMs fully loaded, many blocks queued
    Naive: each SM wastes cycles on strided B cache misses
    Tiled: each SM computes continuously from shared memory
    → Gap between them is large and visible!

Why does N=2048 drop back to 2.2x from 3.0x at N=1024?
    At N=2048, even naive GPU is starting to be more efficient
    because the L2 cache behavior improves at very large N
    (more reuse across the larger working set).
    Also, at N=2048 the tiled T=32 kernel's occupancy may
    be slightly reduced due to register pressure.

---

### Observation 3: GFLOPS Grows with N but Far from Peak

```
Tiled T=32 GFLOPS progression:
N=256:   306.9 GFLOPS  (3.8% of peak)
N=512:   377.4 GFLOPS  (4.6% of peak)
N=1024:  693.7 GFLOPS  (8.5% of peak)
N=2048:  907.7 GFLOPS  (11.1% of peak)

T4 Peak FP32: 8,141 GFLOPS
Best achieved: 907.7 GFLOPS = 11.1% of peak
```

GFLOPS grows with N because:
- More blocks → all 40 SMs fully occupied
- Better SM utilization at larger N
- More work hides memory latency

BUT we're only at 11.1% of peak. Why so far from peak?

**Reason 1: Our kernel doesn't use Tensor Cores**
```
T4 has TWO compute engines:
    CUDA Cores (FP32):    8,141 GFLOPS  ← we use this
    Tensor Cores (FP16):  65,130 GFLOPS ← we DON'T use this

cuBLAS uses Tensor Cores for GEMM.
Our FP32 CUDA core kernel cannot exceed 8,141 GFLOPS,
and we're at 11.1% of that — so we're really at
907.7 / 8141 = 11.1% of CUDA core peak.
```

**Reason 2: Occupancy Limitation**
```
Tiled T=32: 1024 threads/block, 8KB shared mem/block

T4 limits per SM:
    Max threads:       1024
    Max blocks:        16
    Max shared mem:    64KB

With T=32: 64KB / 8KB = 8 blocks per SM max
           8 blocks × 1024 threads = 8192 threads/SM
           T4 max = 1024 threads/SM → only 1 block active!

Wait — T4 max threads per SM = 1024:
    1 block × 1024 threads = 100% thread occupancy
    But only 1 block active → limited by thread count!

Effective occupancy is limited. Fewer active blocks
means less opportunity to hide memory latency with
computation from other warps.
```

**Reason 3: Register Pressure**
```
Tiled T=32 uses:
    - float sum accumulator (1 register)
    - loop variables (k, t, etc.) (~5 registers)
    - address computations (~4 registers)
    Estimated: ~32 registers per thread

T4 has 65,536 registers per SM:
    With 1024 threads: 65536/1024 = 64 registers/thread available
    ~32 used → 50% register utilization
    
Not register-bound, but combined with shared memory
limits, occupancy stays constrained.
```

**Reason 4: No Software Pipelining**
```
Our kernel: LOAD tile → SYNC → COMPUTE → SYNC → next tile
            = serial: compute must WAIT for load to finish

Optimized kernels (cuBLAS, CUTLASS) use double buffering:
    Load tile t+1 while computing tile t
    Hides memory latency behind computation
    
Without double buffering:
    Every tile step: GPU stalls during __syncthreads()
    waiting for slower threads to finish loading
    These stall cycles = wasted compute time
```

**Reason 5: No Vectorized Loads**
```
Our load:
    tile_A[ty][tx] = A[row * N + a_col];  // 1 float = 4 bytes

Optimized kernels use float4 loads:
    // Load 4 floats at once = 16 bytes per instruction
    float4 val = reinterpret_cast<float4*>(&A[...])[0];
    
4× fewer load instructions → 4× better instruction throughput
→ more cycles available for FMA (fused multiply-add)
```

**Summary: why 11.1% and not 80%+?**
```
Gap breakdown (approximate):
    Our best (T=32, N=2048):    907 GFLOPS  (11.1%)
    + Double buffering:        ~1500 GFLOPS (~18%)
    + Vectorized loads:        ~2500 GFLOPS (~31%)
    + Better occupancy tuning: ~3500 GFLOPS (~43%)
    + Tensor Cores (FP16):    ~6500 GFLOPS (~80%)
    cuBLAS achieves:          ~6500 GFLOPS (~80% of Tensor Core peak)

Our simple tiled kernel is a correct foundation.
Production GEMM requires 6+ advanced optimizations on top.
This is exactly why cuBLAS exists as a library!
```

---

### Observation 4: T=32 vs T=16 — When Does Larger Tile Win?

```
N=256:  T=16: 302.3 GFLOPS   T=32: 306.9 GFLOPS  → T=32 barely wins (1.5%)
N=512:  T=16: 356.2 GFLOPS   T=32: 377.4 GFLOPS  → T=32 wins (5.9%)
N=1024: T=16: 370.3 GFLOPS   T=32: 693.7 GFLOPS  → T=32 wins strongly (87%)
N=2048: T=16: 660.4 GFLOPS   T=32: 907.7 GFLOPS  → T=32 wins (37%)
```

At N=256: barely any difference
```
Only 256/32 × 256/32 = 64 blocks for T=32
Only 256/16 × 256/16 = 256 blocks for T=16
At small N, T=16 actually has more blocks → better SM occupancy
T=32's reuse advantage is canceled by occupancy disadvantage
Net result: nearly identical
```

At N=1024: T=32 is 1.87x faster than T=16
```
T=16: 1024/16 × 1024/16 = 4096 blocks
T=32: 1024/32 × 1024/32 = 1024 blocks
Both have enough blocks to saturate 40 SMs
T=32's 2× reuse advantage NOW dominates
→ 87% speedup from T=16 to T=32
```

Key insight: tile size optimization is N-dependent!
For deployment, benchmark at your actual working N.

---

### Observation 5: CPU Catastrophic at Large N

```
N=256:   CPU = 22ms      GPU T=32 = 0.11ms   → 202x speedup
N=512:   CPU = 282ms     GPU T=32 = 0.71ms   → 396x speedup
N=1024:  CPU = 3240ms    GPU T=32 = 3.10ms   → 1046x speedup
N=2048:  CPU = ~81,212ms GPU T=32 = 18.93ms  → ~4291x speedup (est.)
```

CPU scaling: O(N³) — each 2× in N = 8× more time
```
N=256→512:  CPU 12.7× slower,  GPU 4.4× slower
N=512→1024: CPU 11.5× slower,  GPU 2.7× slower
N=1024→2048: CPU ~25× slower,  GPU 4.5× slower
```

CPU scales with O(N³) as expected.
GPU scales much better because parallelism grows with N²
(more output elements = more threads) while compute scales N³.
The GPU's parallelism absorbs the N growth far better.

At N=2048: CPU takes 81 SECONDS for one matrix multiply!
A real-time application needs this in <1ms per frame.
This is WHY GPU is non-negotiable for neural networks.
(ResNet-50 inference = thousands of matrix multiplies per image!)

---

### Observation 6: Effective Bandwidth Drops as N Grows

```
Tiled T=32 effective bandwidth (= 3N²×4 / time):
N=256:   7.2 GB/s
N=512:   4.4 GB/s
N=1024:  4.1 GB/s
N=2048:  2.7 GB/s
```

This seems counterintuitive — bandwidth DECREASES as N grows?

```
Effective bandwidth = minimum bytes (3N²×4) / time
Time grows faster than N² because of O(N³) FLOPs.

For large N: kernel is deeply compute-bound.
Time is dominated by FLOPs, not memory.
Memory accesses complete quickly (reused from shared mem),
but FLOPs take proportionally longer.

So: low bandwidth = not because memory is slow,
    but because compute is the bottleneck.
    Shared memory is doing its job — DRAM barely used!

Compare to naive GPU:
    N=1024 Naive: 1.4 GB/s bandwidth
    N=1024 T=32:  4.1 GB/s bandwidth

Tiled has HIGHER effective bandwidth than naive despite
using shared memory reuse!

Why? Because naive wastes so much time on cache misses
(strided B access) that it spends proportionally MORE
time per byte of useful output data produced.
Tiled computes faster → more useful bytes per second.
```

---

## Hypothesis Validation

| Hypothesis | Prediction | Actual | Verdict |
|-----------|-----------|--------|---------|
| Tiled vs naive speedup ~10-16x | 10-16x | 1.5-3.0x | ⚠️ Much lower — explained below |
| T=32 slightly faster than T=16 | small margin | 1.5-1.87x (larger than expected) | ⚠️ T=32 wins more than predicted |
| Tiled achieves 30-60% of peak GFLOPS | 30-60% | 11.1% | ❌ Much lower — Tensor Cores not used |
| GPU tiled 50-200x faster than CPU | 50-200x | 202-1046x | ✅ Confirmed (exceeded upper bound!) |
| Naive GPU beats CPU | 5-20x | 138-352x | ✅ Confirmed (exceeded prediction!) |
| Transfers dominate less than Exp 1 | Yes | 3.4% vs 97.5% in Exp 1 | ✅ Strongly confirmed |
| Naive bandwidth lower than vector add | Yes | 1.4 GB/s vs 260 GB/s | ✅ Confirmed |

**On tiled vs naive being only 1.5-3x, not 10-16x:**
```
Prediction assumed: speedup ≈ TILE_SIZE (16× for T=16)
Reality: 1.5-3x

Why wrong?

The T× reduction in DRAM traffic does NOT translate to
T× speedup because at N=1024 even naive matmul is
already compute-bound (170 FLOP/byte >> 25.4 ridge point).

A compute-bound kernel's time is dominated by FLOPs, not memory.
Reducing memory traffic doesn't help if compute is the bottleneck!

Tiled wins by:
    1. Fixing strided B access (coalesced loads to shared mem)
    2. Reducing L2 cache pressure → other warps get more bandwidth
    3. More predictable execution → less variance

NOT primarily by reducing DRAM traffic (already cached in L2 for naive).

This is a critical insight:
    Tiling helps MOST when the kernel is memory-bound.
    For compute-bound kernels, the benefit is more modest.
    The REAL bottleneck for our kernel is FLOPs per SM,
    not DRAM bandwidth.
```

---

## Why Only 11.1% of Peak GFLOPS — Summary

| Gap Source | Impact | Fix |
|-----------|--------|-----|
| Not using Tensor Cores | 8× lower ceiling (FP32 vs FP16 Tensor) | Use FP16 + WMMA |
| No double buffering | Stalls during tile loads | Prefetch tile t+1 while computing t |
| No vectorized loads | 4× more load instructions | Use float4 loads |
| Occupancy limits (T=32: 1 block/SM) | Less latency hiding | Tune block size |
| No register tiling | Low ILP per thread | Each thread compute 4×4 output tile |
| Single accumulator | Sequential FMA chain | Unroll + multiple accumulators |

cuBLAS achieves ~80% by addressing ALL of the above simultaneously.
Our kernel addresses NONE of them — it is a clean pedagogical baseline.

---

## nvprof Metrics Limitation

nvprof hardware metrics not collectible for CC 7.5+ (Tesla T4).
Theoretical values based on kernel analysis:

| Metric | Naive (Expected) | Tiled T=32 (Expected) | Reasoning |
|--------|-----------------|----------------------|-----------|
| achieved_occupancy | ~60-70% | ~50-60% | T=32: 1024 threads fills SM thread limit |
| gld_efficiency | ~50-70% | ~95-100% | Naive: strided B; Tiled: coalesced loads |
| gst_efficiency | ~99% | ~99% | Both write C sequentially |
| L2 hit rate | ~40-60% | ~80-90% | Tiled reuses tiles from shared, less L2 pressure |
| shared_efficiency | N/A | ~85-95% | Shared mem well utilized in tiled |

Will attempt ncu (Nsight Compute) for Experiment 3 to validate.

---

## Questions for Experiment 3 (Parallel Reduction)

1. **Reduction is fundamentally different**: unlike matmul where each thread owns one output element, reduction requires ALL threads to contribute to ONE output value. How does this change the synchronization model?

2. **Tree reduction**: halving active threads each step — does warp divergence hurt performance? At what step does it become a problem?

3. **Warp shuffle instructions**: `__shfl_down_sync()` eliminates shared memory for final warp reduction. How much faster is this vs shared memory approach?

4. **Atomic operations**: when to use `atomicAdd` vs tree reduction? What is the performance cost of atomics under high contention?

5. **Occupancy vs work per thread**: reduction benefits from high occupancy (many warps to hide latency). Does this conflict with using more registers per thread for ILP?

6. **Comparison to thrust::reduce**: how close can a hand-written reduction get to the highly optimized Thrust library implementation?
