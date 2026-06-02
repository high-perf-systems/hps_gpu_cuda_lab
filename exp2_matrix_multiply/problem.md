# Experiment 2: Matrix Multiplication

## Problem Statement

Given two square matrices A and B of size N×N,
compute C = A × B where:

    C[row][col] = sum over k of A[row][k] * B[k][col]

Each output element C[row][col] requires N multiply-add
operations — one for each element along the shared dimension.

This is the canonical GPU optimization problem because it
exposes a fundamental architectural tension: the ratio of
compute to memory access grows with N, making it possible
to hide memory latency behind computation — something
vector addition (Experiment 1) could never do.

---

## Motivation

Experiment 1 (vector addition) showed us the worst case
for GPU utilization: a purely memory-bound kernel where
every byte loaded is used for exactly one arithmetic operation.
We achieved ~80% of peak bandwidth but the kernel itself
was only 2.5% of total execution time.

Matrix multiplication is the opposite extreme:

    Vector addition:   1 FLOP  per 2 memory reads  (ratio 0.5)
    Matrix multiply:   2 FLOPS per 2 memory reads, BUT
                       each element reused N times  (ratio N)

As N grows, the arithmetic intensity grows proportionally.
This is why matrix multiplication is the first problem where
shared memory optimization produces dramatic, measurable speedup.

Real-world relevance:
    → Neural network training: weight matrix × activation matrix
    → Computer graphics: transformation matrix chains
    → Signal processing: frequency domain transforms
    → Sensor fusion : projection matrices

---

## Core Concepts Being Tested

### Concept 1: Arithmetic Intensity

Arithmetic intensity = FLOPs / bytes_moved_from_DRAM

For the entire N×N matrix computation:

    Vector addition:
        FLOPs:     2N
        DRAM:      12N bytes
        Intensity: 1/6 FLOP/byte (constant, memory-bound)

    Naive matmul:
        FLOPs:     2N³
        DRAM:      12N² bytes (each of 3N² elements read once)
        Intensity: N/6 FLOP/byte (grows with N!)

    Tiled matmul (T×T tile):
        FLOPs:     2N³ (UNCHANGED — tiling doesn't skip work!)
        DRAM:      reduced vs naive by approximately T× for B
                   (tiling primarily fixes B's strided access)
        Intensity: higher than naive by ~T×

Key insight:
    Tiling does NOT reduce FLOPs.
    Tiling fixes TWO problems:
        1. Converts strided B access to coalesced DRAM loads
        2. Reuses loaded data T times from fast shared memory
           instead of re-fetching from slow global memory

T4 ridge point = 8,100 GFLOPS / 320 GB/s = 25.3 FLOP/byte

At N=1024:
    Naive intensity:    ~170 FLOP/byte → should be compute-bound
    Tiled T=16:         ~2,730 FLOP/byte → even more compute-bound

BUT naive's poor access pattern means effective intensity is
much lower than theoretical. The actual speedup from tiling
will tell us how much the access pattern was hurting performance.

T4 ridge point = 8100 GFLOPS / 320 GB/s = 25.3 FLOP/byte

ALL versions of matmul (even naive) are above the ridge point
at N=1024 — theoretically compute-bound, not memory-bound!

BUT naive matmul's poor access pattern (strided B access)
means it WASTES cache lines and effective intensity is lower.
Tiling makes the theoretical intensity actually achievable.

Higher arithmetic intensity = GPU can hide memory latency
by keeping compute units busy between memory requests.

### Concept 2: Memory Coalescing in 2D (Naive Version)

In a naive matmul with row-major storage:

Thread (row, col) computes C[row][col]:
    Reads A[row][0..N-1]   → consecutive in memory ✅ coalesced
    Reads B[0..N-1][col]   → strided by N in memory ❌ NOT coalesced

When a warp of 32 threads (same row, adjacent cols) reads B:
    Thread 0  reads B[k][0], B[k][1]... B[k][0]
    Thread 1  reads B[k][1]
    Thread 31 reads B[k][31]

Wait — these ARE adjacent! So B reads ARE coalesced for
the inner loop step at fixed k!

But across iterations (k=0,1,2...):
    Thread 0 reads B[0][col], B[1][col], B[2][col]...
    Stride = N between consecutive k values
    For N=1024: stride = 4096 bytes → ONE element per cache line!
    Each k-step = 32 separate cache line fetches for a warp

This strided access pattern is what shared memory fixes.

### Concept 3: Shared Memory Tiling

Key insight: if we load a TILE of A and B into shared memory,
all TILE_SIZE threads in a row/column can reuse the same data.

Tiling algorithm:
    Divide A and B into tiles of size TILE_SIZE × TILE_SIZE
    For each tile pair (t):
        1. Collaboratively load tile of A into shared mem
        2. Collaboratively load tile of B into shared mem
        3. __syncthreads()   ← wait for ALL threads to finish loading
        4. Each thread computes partial dot product from shared tiles
        5. __syncthreads()   ← wait before loading next tile
    Accumulate partial sums → final C[row][col]

DRAM reads per output element:
    Naive:  2N global memory reads
    Tiled:  2N/TILE_SIZE global memory reads
    Reduction factor: TILE_SIZE (16x or 32x!)

### Concept 4: __syncthreads() is Critical Here

Without __syncthreads() after loading tiles:
    Thread 0 might start computing with tile data
    before Thread 31 has finished loading its tile elements!
    → Data race on shared memory → wrong results!

This is MORE critical than in vector addition because:
    → Threads DEPEND on each other's shared memory writes
    → Missing sync = silent wrong answers (not crash!)
    → Very hard to debug without knowing this pattern

### Concept 5: Tile Size Tradeoffs

TILE_SIZE = 16 (256 threads per block):
    Shared memory per block: 2 × 16×16 × 4 bytes = 2 KB
    Threads per block: 256
    Max blocks per SM: limited by shared memory
    T4: 64KB shared mem / 2KB = 32 blocks possible
    Data reuse factor: 16×

TILE_SIZE = 32 (1024 threads per block):
    Shared memory per block: 2 × 32×32 × 4 bytes = 8 KB
    Threads per block: 1024 (maximum per block on T4!)
    Max blocks per SM: 64KB / 8KB = 8 blocks possible
    Data reuse factor: 32×

Larger tile = more reuse, but:
    → More shared memory per block → fewer concurrent blocks
    → More threads per block → register pressure increases
    → Occupancy may go DOWN even as reuse goes UP

Optimal tile size depends on register count, shared memory
size, and the specific GPU's SM configuration.
We test BOTH to measure which wins on T4.

### Concept 6: Roofline Model

The Roofline model predicts performance limits:

    If arithmetic_intensity < ridge_point:
        Performance limited by MEMORY BANDWIDTH
        Max performance = bandwidth × arithmetic_intensity

    If arithmetic_intensity > ridge_point:
        Performance limited by COMPUTE (FLOPs)
        Max performance = peak_FLOPS

T4 specs:
    Peak bandwidth:    320 GB/s
    Peak FP32 compute: 8.1 TFLOPS

Ridge point = peak_FLOPS / peak_bandwidth
            = 8,100 GFLOPS / 320 GB/s
            = 25.3 FLOP/byte

Vector addition arithmetic intensity: 0.083 FLOP/byte
    → Deep in memory-bound region

Naive matmul at N=1024: ~256 FLOP/byte
    → Far into compute-bound region (theoretically!)
    → But poor access patterns may reduce effective intensity

Tiled matmul at N=1024, T=16: ~4096 FLOP/byte
    → Extremely compute-bound
    → Performance ceiling is peak_FLOPS, not bandwidth!

We will measure actual FLOPS achieved and compare to peak.

---

## Versions to Implement

### Version 1: CPU Baseline
Sequential triple-nested loop.
Standard O(N³) algorithm.
Purpose: correctness reference + baseline comparison.

### Version 2: GPU Naive (Global Memory Only)
One thread per output element C[row][col].
Reads A row-wise (coalesced) and B column-wise (strided).
No shared memory, no tiling.
Purpose: shows baseline GPU performance, quantifies
the cost of unoptimized memory access patterns.

### Version 3: GPU Tiled (TILE_SIZE = 16)
Shared memory tiling with 16×16 tiles.
256 threads per block.
16× reduction in global memory traffic vs naive.
Purpose: primary optimization demonstration.

### Version 4: GPU Tiled (TILE_SIZE = 32)
Shared memory tiling with 32×32 tiles.
1024 threads per block (maximum for T4).
32× reduction in global memory traffic vs naive.
Purpose: compare against TILE_SIZE=16, find optimal tile.

---

## Matrix Sizes to Test

We will test square matrices for three values of N:

| N | Matrix Size | Memory per Matrix | Total (A+B+C) |
|---|------------|------------------|---------------|
| 256 | 256×256 | 0.25 MB | 0.75 MB |
| 512 | 512×512 | 1.00 MB | 3.00 MB |
| 1024 | 1024×1024 | 4.00 MB | 12.00 MB |
| 2048 | 2048×2048 | 16.00 MB | 48.00 MB |

Note: CPU baseline becomes very slow at N=2048 (O(N³)).
      We will time-limit CPU runs for large N.

---

## Performance Metric: GFLOPS

Unlike vector addition (measured in GB/s bandwidth),
matrix multiply is best measured in GFLOPS:

    FLOPs per multiply-add = 2 (one multiply + one add)
    Total FLOPs = 2 × N³

    GFLOPS = (2 × N³) / (time_seconds × 1e9)

This normalizes across matrix sizes and lets us compare
against T4 peak (8.1 TFLOPS = 8,100 GFLOPS for FP32).

We will report BOTH GB/s bandwidth AND GFLOPS.

---

## Hypothesis

### Q1: How much faster is tiled vs naive GPU?
Tiled reduces global memory traffic by TILE_SIZE factor.
Memory access time should reduce by ~TILE_SIZE.
Prediction: tiled T=16 is ~10-16x faster than naive.
(Not exactly 16x because compute time also present)

### Q2: TILE_SIZE=16 vs TILE_SIZE=32: which wins?
TILE_SIZE=32 has 2x more reuse but 4x more shared memory.
Occupancy may drop due to shared memory limits.
Prediction: TILE_SIZE=32 slightly faster due to better reuse,
but not 2x faster (occupancy tradeoff prevents full benefit).

### Q3: How close to T4 peak FLOPS?
T4 peak FP32: 8,100 GFLOPS
Tiled matmul is compute-bound at large N.
Prediction: tiled achieves 30-60% of peak FLOPS
(cuBLAS achieves ~80%+ with advanced optimizations).
Our simple tiled kernel will be a good but not optimal implementation.

### Q4: CPU vs GPU speedup?
CPU single-threaded matmul is O(N³) with poor cache behavior.
GPU with tiling should show massive speedup.
Prediction: 50-200x GPU speedup at N=1024 (tiled vs CPU).

### Q5: Does naive GPU beat CPU?
Even without optimization, GPU parallelism helps.
Prediction: naive GPU still 5-20x faster than CPU
(parallelism benefit > poor access pattern cost).

### Q6: Memory bandwidth vs Experiment 1?
Naive matmul has poor B-matrix access pattern.
Tiled matmul reduces DRAM bandwidth dramatically.
Prediction:
    Naive: lower effective bandwidth than vector add
           (strided B access wastes cache lines)
    Tiled: much lower DRAM bandwidth (data reused from shared mem)
           but much higher compute throughput

---

## What Success Looks Like

After this experiment I should be able to:
    ✅ Explain why naive matmul has poor memory access for B
    ✅ Draw the tiling diagram and explain __syncthreads() placement
    ✅ Calculate arithmetic intensity for each version
    ✅ Explain TILE_SIZE tradeoffs (reuse vs occupancy)
    ✅ Explain the roofline model and where each version sits
    ✅ Quantify speedup: tiled vs naive vs CPU
    ✅ Connect to cuBLAS and why it achieves higher GFLOPS
    ✅ Explain why shared memory matters for the rasterizer
