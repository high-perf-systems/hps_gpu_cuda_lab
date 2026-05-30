# Experiment 1: Vector Addition

## Problem Statement

Given two arrays A and B of N floating-point numbers,
compute C = A + B element-wise.

This is trivially parallelizable: each output element
C[i] = A[i] + B[i] is completely independent of every
other element. No thread needs to communicate with any
other thread. This makes it a perfect first GPU kernel.

## Motivation

Vector addition is the simplest possible GPU program,
but it teaches the COMPLETE CUDA programming model:

1. How to allocate memory on GPU (cudaMalloc)
2. How to transfer data CPU → GPU (cudaMemcpy H2D)
3. How to write a kernel (__global__ function)
4. How to launch a kernel (<<<grid, block>>> syntax)
5. How to transfer results GPU → CPU (cudaMemcpy D2H)
6. How to free GPU memory (cudaFree)
7. How to profile GPU code (nvprof)

Every future CUDA program follows this SAME structure.
Master this, and the pattern is always the same.

## Core Concepts Being Tested

### Concept 1: Thread Indexing
Each thread computes ONE element of C.
Thread must know: "which index am I responsible for?"

Global thread ID = blockIdx.x * blockDim.x + threadIdx.x

This is the single most important formula in CUDA.

### Concept 2: Memory Hierarchy
Data must explicitly move between CPU and GPU.
This transfer cost is real and measurable!

### Concept 3: Grid/Block/Thread Organization
We must launch ENOUGH blocks to cover ALL N elements.
Formula: num_blocks = ceil(N / threads_per_block)
       = (N + threads_per_block - 1) / threads_per_block

### Concept 4: Boundary Guard
If N is not a multiple of block size, the last block
has some threads with no work to do. Without a guard:
    → Those threads access out-of-bounds memory
    → Undefined behavior / crash

Solution: if (idx < N) return; — always required!

### Concept 5: Memory Bandwidth
Vector addition is MEMORY BOUND, not COMPUTE BOUND.

Each element: 2 reads (A[i], B[i]) + 1 write (C[i])
              1 addition (trivial compute)

The GPU is just moving data — arithmetic is free!
This teaches us: bottleneck is memory bandwidth,
not floating-point throughput.

We will MEASURE this with nvprof!

## Versions to Implement

### Version 1: Basic (CPU baseline)
Sequential CPU implementation for correctness check
and baseline performance comparison.

### Version 2: Naive GPU
One thread per element, global memory only.
No optimizations. Just the pure CUDA pattern.

### Version 3: Pinned Memory GPU
Same kernel, but use cudaMallocHost for CPU arrays
instead of regular malloc/new.

Hypothesis: pinned memory should improve H2D/D2H
transfer speed because DMA can work directly.

## Hypothesis

### Q1: How much faster will GPU be vs CPU?
For large N (e.g., 1M-16M elements):
GPU should be significantly faster due to parallelism.

But for small N (e.g., 1K elements):
GPU might be SLOWER than CPU due to launch overhead
and memory transfer cost.

Prediction: GPU wins only when N is large enough that
parallelism benefit > transfer overhead.

### Q2: Where will time be spent on GPU?
My prediction: H2D + D2H transfers will dominate
over actual kernel execution time.

Reason: Vector addition is memory-bandwidth bound.
The kernel itself is trivial (one addition per thread).
But we must transfer data TO and FROM GPU.

### Q3: Pinned vs pageable memory?
Pinned memory should be faster for transfers.
Prediction: 20-40% faster H2D/D2H with pinned memory.

### Q4: What will be the memory bandwidth?
Tesla T4 theoretical peak: ~320 GB/s
Expected actual: 60-80% of peak = ~190-250 GB/s
(Real systems rarely achieve theoretical peak)

## What Success Looks Like

After this experiment I should be able to:
    ✅ Write a complete CUDA program from scratch
    ✅ Explain every line of code confidently
    ✅ Explain the thread indexing formula
    ✅ Explain why we need boundary guards
    ✅ Read nvprof output and identify bottlenecks
    ✅ Explain pinned vs pageable memory tradeoff
    ✅ Explain when GPU is faster/slower than CPU.


