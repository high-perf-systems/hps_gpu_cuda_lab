# Experiment 3: Parallel Reduction on GPU

## Problem

Given an array A of N floating-point elements, compute a single scalar result by
repeatedly applying an associative binary operation across all elements:

```
result = A[0] ⊕ A[1] ⊕ A[2] ⊕ ... ⊕ A[N-1]
```

For this experiment the operation is addition — the reduction sum:

```
result = Σ A[i],  i = 0 .. N-1
```

On a CPU this is trivial: one loop, one accumulator, O(N) time on one core.
On a GPU with thousands of threads, the challenge is fundamentally different —
every thread must contribute to a single scalar output, which requires careful
coordination. Getting this coordination wrong produces either incorrect results
(data races) or correct but catastrophically slow results (serialised execution).

---

## Connection to Previous Experiments

Exp1 (vector add) and exp2 (matrix multiply) were **embarrassingly parallel** —
each output element was computed independently by exactly one thread. No thread
needed to communicate with any other thread. Reduction breaks this completely.

In reduction, every thread in the grid must eventually contribute to the same
single output value. This introduces two problems that did not exist before:

**Problem 1 — Synchronisation:** threads must coordinate writes to shared memory
locations. Without synchronisation, threads read stale values written by other
threads — a data race.

**Problem 2 — Parallelism degrades as the problem converges:** at the start of
reduction, all N threads are active. After one step, N/2 threads are active.
After two steps, N/4. By the final step, only one thread is working. The GPU's
massive parallelism advantage diminishes precisely as the computation converges.
How you manage this degradation determines how fast the kernel runs.

---

## Arithmetic Intensity and Expected Bottleneck

Following the roofline framework established in exp2:

```
Reduction sum:
    FLOPs per element   :  O(log N) adds  ≈  20 adds for N=1M
    Bytes per element   :  4 bytes (one float read, essentially once)
    Arithmetic intensity:  ~20 / 4  =  5 FLOP/byte

T4 Ridge Point          :  8141 GFLOPS / 320 GB/s  =  25.4 FLOP/byte

Since 5 FLOP/byte << 25.4 FLOP/byte → reduction is MEMORY BOUND
```

This places reduction in the same regime as vector addition from exp1 — the
bottleneck is how fast we can move data from DRAM to the SMs, not how fast
we can compute. The goal is therefore to maximise memory throughput and
minimise wasted bandwidth — not to increase arithmetic intensity.

---

## Core Concepts

### 1. The Tree Reduction Idea

The naive sequential approach takes O(N) steps. The parallel approach uses a
**binary tree** reduction to take O(log N) steps:

```
Step 0:  [1, 2, 3, 4, 5, 6, 7, 8]   — N=8 elements, 8 threads active
Step 1:  [3, _, 7, _, 11, _, 15, _]  — 4 threads add pairs
Step 2:  [10, _, _, _, 26, _, _, _]  — 2 threads add step-1 results
Step 3:  [36, _, _, _, _, _, _, _]   — 1 thread adds step-2 results
Result:  36
```

Each step halves the active thread count and halves the remaining work. After
log₂(N) steps only one thread holds the final result. The total work done is
N/2 + N/4 + N/8 + ... + 1 = N-1 additions — identical to the sequential case,
but distributed across threads and completed in log₂(N) parallel steps.

### 2. __syncthreads() — The Block-Level Barrier

Within a CUDA thread block, threads can communicate through **shared memory** —
a fast on-chip SRAM that all threads in the block can read and write. But shared
memory writes are not instantaneously visible to other threads. Without explicit
synchronisation, thread A might read a shared memory location before thread B
has finished writing to it — a data race.

`__syncthreads()` is a **barrier**: every thread in the block must reach this
call before any thread is allowed to proceed past it. It guarantees that all
shared memory writes issued before the barrier are visible to all threads after
the barrier.

```cpp
// Without __syncthreads() — DATA RACE
sdata[tid] = A[tid];          // thread 0 writes
float val = sdata[tid + 1];   // thread 0 reads — might see stale value!

// With __syncthreads() — CORRECT
sdata[tid] = A[tid];
__syncthreads();              // all threads have written before any reads
float val = sdata[tid + 1];   // safe — all writes are visible
```

Cost: `__syncthreads()` stalls all active warps in the block until the slowest
thread arrives. It serialises the block at each barrier point. Every barrier
is a potential stall — minimising barrier calls is a key optimisation target.

### 3. Warp Divergence

The GPU executes threads in groups of 32 called **warps**. All 32 threads in a
warp execute the same instruction simultaneously — this is the GPU's fundamental
SIMD execution model. When threads in the same warp take different branches of
an if/else, the hardware must execute both branches serially, masking threads
that did not take each path. This is **warp divergence**.

```cpp
// Naive reduction with interleaved addressing — causes warp divergence
if (tid % (2 * s) == 0) {     // only even-strided threads are active
    sdata[tid] += sdata[tid + s];
}
```

In the first step (s=1), threads 0, 2, 4, 6... are active and threads 1, 3, 5...
are idle. Within each warp of 32 threads, 16 are active and 16 are idle. The
hardware executes the active thread path (with idles masked), then the idle
thread path (which does nothing, with actives masked). The warp effectively runs
at half throughput — 16 useful operations out of 32 execution slots.

In later steps the divergence gets worse: step 2 has only 8 active threads per
warp, step 3 has 4, and so on. Throughput halves at every step.

**The fix — sequential addressing:**

```cpp
// Sequential addressing — avoids divergence within a warp
int idx = 2 * s * tid;
if (idx < blockDim.x) {
    sdata[idx] += sdata[idx + s];
}
```

With sequential addressing, the first s threads are active and the remaining
threads are idle. Within each warp, either all 32 threads are active (warps
at the start of the block) or all 32 are idle (warps at the end). No warp
has a mixed active/idle split — divergence is eliminated for all but the
boundary warp. Throughput approaches 32 useful operations per warp.

### 4. Shared Memory Bank Conflicts

Shared memory on the T4 is divided into 32 **banks** of 4-byte words. When
multiple threads in a warp access different addresses that map to the same bank,
the accesses are serialised — a **bank conflict**. A k-way bank conflict means
the access takes k times as long as a single access.

Address → bank mapping:
```
bank_number = (address / 4) % 32
```

Two threads access the same bank if:
```
(address_A / 4) % 32 == (address_B / 4) % 32
```

Example of a 2-way bank conflict in sequential addressing:
```
Step s=16: thread 0 accesses sdata[0] and sdata[16]
           thread 1 accesses sdata[2] and sdata[18]
           thread 16 accesses sdata[32] and sdata[48]

sdata[0]  → bank 0
sdata[16] → bank 16    (no conflict here)

sdata[0]  → bank 0
sdata[32] → bank 0     (thread 0 and thread 16 both hit bank 0 → 2-way conflict!)
```

In the final steps of reduction when only a few threads remain, bank conflicts
can serialise what should be parallel accesses. The fix for the last warp is to
use warp shuffle instructions instead of shared memory — eliminating shared
memory access entirely for the last 32 threads.

### 5. Warp Shuffle Instructions

`__shfl_down_sync()` is a hardware instruction that allows threads within a warp
to directly exchange register values **without going through shared memory**. No
shared memory allocation, no bank conflicts, no `__syncthreads()` needed within
a warp (threads in the same warp are implicitly synchronised at the instruction
level).

```cpp
// Warp-level reduction using shuffle — replaces shared memory for last 32 threads
__device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

`__shfl_down_sync(mask, val, offset)` — thread T receives the value of val from
thread T+offset within the same warp. Thread 0 receives thread 16's value, then
thread 8's, then thread 4's, then thread 2's, then thread 1's. After 5 shuffles,
thread 0 holds the sum of all 32 threads — without a single shared memory access.

The `0xffffffff` mask means all 32 threads in the warp participate.

This is the GPU equivalent of the CPU's SIMD horizontal add — operating across
lanes of a vector register without going through memory.

### 6. The Optimisation Progression

The full reduction optimisation journey follows NVIDIA's canonical progression:

```
Version 1: Interleaved addressing
           — warp divergence, 50% throughput loss per step
           
Version 2: Sequential addressing
           — divergence eliminated, but bank conflicts in final steps
           
Version 3: First add during load
           — each thread loads two elements and adds them before entering
             the reduction loop, halving the number of required steps
             
Version 4: Unroll last warp
           — when only 32 threads remain, remove __syncthreads() calls
             (warp is implicitly synchronised) and unroll the loop
             
Version 5: Warp shuffle
           — replace shared memory in the last warp entirely with
             __shfl_down_sync(), eliminating bank conflicts and barriers
             
Version 6: Compare against thrust::reduce
           — how close does our best kernel get to the Thrust library?
```

Each version fixes exactly one bottleneck. Measuring each version separately
isolates the contribution of each optimisation — the same hypothesis-first
methodology used throughout this lab series.

---

## Experimental Approach

A CPU baseline is measured first — single-threaded sequential reduction using
`std::accumulate`. This establishes the correctness reference (all GPU results
must match) and the performance floor.

Six GPU kernel versions are implemented in order, each building on the previous.
For each version:
- Correctness verified against CPU result
- Kernel time measured (warmup + 10 timed runs, average reported)
- Bandwidth computed: `2 × N × sizeof(float) / time` (read all N, write 1)
- % of T4 peak bandwidth (320 GB/s) computed
- ncu used to measure achieved occupancy and memory throughput

The sweep covers N = {1M, 4M, 16M, 64M} to observe how each version scales.

A final comparison against `thrust::reduce` establishes the gap between our
best hand-written kernel and the highly optimised library implementation.

---

## Scope

- Reduction sum only (operation = addition)
- Single kernel launch per reduction (no multi-pass for very large N)
- Single block reduction for simplicity in versions 1-4; grid-level reduction
  in versions 5-6 using atomic add for the final inter-block accumulation
- T4 GPU (Compute Capability 7.5), Google Colab
- nvcc -O2, CUDA 12.x
- ncu for hardware metrics where available

---

## Questions This Experiment Will Answer

1. How much does warp divergence cost in practice — is it measurable?
2. How much does sequential addressing (no divergence) improve over
   interleaved addressing?
3. At what step count does unrolling the last warp become measurable?
4. How much faster is warp shuffle vs shared memory for the final warp?
5. What fraction of T4 peak memory bandwidth does the best kernel achieve?
6. How close does our best kernel get to thrust::reduce?