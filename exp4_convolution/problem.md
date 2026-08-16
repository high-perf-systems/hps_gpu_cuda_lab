# Experiment 4: 2D Convolution on GPU

## Problem Statement

Given a single-channel (grayscale) input image of size H×W and a convolution
kernel of size Kh×Kw, compute the output feature map O where:

```
O[row][col] = Σ_i Σ_j  Input[row+i][col+j] × Kernel[i][j]
              i = 0..Kh-1,  j = 0..Kw-1
```

Boundary condition: zero-padding — any input index outside [0, H) × [0, W)
is treated as 0. With padding=0 (no explicit padding added), the output
dimensions are:

```
out_H = H - Kh + 1
out_W = W - Kw + 1
```

Each output element requires Kh×Kw multiply-accumulate operations.
Total FLOPs for the full convolution: 2 × Kh × Kw × out_H × out_W.

We use square images and square kernels throughout: H=W=N, Kh=Kw=K.

---

## Connection to Previous Experiments

| Experiment | Operation | Independent outputs? | Memory pattern | Bottleneck |
|------------|-----------|---------------------|----------------|------------|
| Exp 1: Vec add | C[i] = A[i]+B[i] | ✅ yes | 1 read → 1 write | Bandwidth |
| Exp 2: Matmul | C[r][c] = dot(A_row, B_col) | ✅ yes | each input reused N times | Compute (large N) |
| Exp 3: Reduction | result = Σ A[i] | ❌ must communicate | tree of writes | Bandwidth |
| **Exp 4: Conv2D** | **O[r][c] = sum over K×K patch** | **✅ yes** | **each input reused up to K² times** | **Bandwidth (small K)** |

Convolution sits between matmul and reduction:
- Like matmul, outputs are independent — no inter-thread communication needed.
- Like matmul, the input is **reused**: each input pixel contributes to up to K²
  output pixels. Tiling can exploit this reuse.
- Unlike matmul, arithmetic intensity grows with **K** (kernel size), not with N
  (image size). For the small kernels used in CNNs (3×3, 5×5), even the best tiled
  kernel remains **memory-bound**.

This makes convolution the most practically important GPU kernel in deep learning —
and the one where hardware-specific tricks (constant memory, shared memory tiling
with halos) matter most.

---

## Arithmetic Intensity

Following the roofline framework from exp2 and exp3:

```
Convolution arithmetic intensity (tiled, each input pixel loaded once):

    FLOPs per output pixel    =  2 × K²
    Bytes per input pixel     =  4  (float32)
    Bytes per output pixel    =  4  (float32)
    Minimum bytes moved       ≈  (N² + out_N²) × 4  ≈  2 × N² × 4  (for small K)

    Arithmetic intensity      ≈  (2 × K² × out_N²) / (2 × N² × 4)
                              ≈  K² / 4   FLOP/byte   (for large N where out_N ≈ N)

T4 ridge point  =  8141 GFLOPS / 320 GB/s  =  25.4 FLOP/byte

K=3  →  intensity ≈  2.25 FLOP/byte   → deep in memory-bound region
K=5  →  intensity ≈  6.25 FLOP/byte   → still memory-bound
K=7  →  intensity ≈  12.25 FLOP/byte  → still memory-bound
K=11 →  intensity ≈  30.25 FLOP/byte  → crosses into compute-bound!
```

Key insight: **for every kernel size used in modern CNNs (3×3 to 7×7),
convolution is memory-bound**. The goal is to maximise effective memory
bandwidth — the same objective as exp1 (vector add) and exp3 (reduction),
but now with spatial reuse as the lever.

Naive (global memory only): each output pixel independently reads its K×K
input patch. The same input pixel is re-fetched from DRAM up to K² times by
different threads. Effective bandwidth is wasted on redundant DRAM traffic.

Tiled (shared memory): an entire block collaboratively loads a tile of the
input (including halos) once, then all threads compute from fast shared memory.
Each input pixel is fetched from DRAM exactly once per block — eliminating
the redundant fetches.

---

## Core Concepts

### Concept 1: 2D Thread Indexing

Each thread is responsible for exactly one output pixel O[row][col].

```python
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

Grid dimensions:
```python
grid_x = ceil(out_W / BLOCK_W)   # blocks along width
grid_y = ceil(out_H / BLOCK_H)   # blocks along height
```

Boundary guard: threads with row >= out_H or col >= out_W must exit immediately
(when out_H or out_W is not a multiple of the block size).

This is the 2D generalisation of the thread index formula from exp1. Every
future 2D image kernel uses this exact pattern.

### Concept 2: The Kernel is Small and Read-Only — Constant Memory

The convolution kernel (the weights, not the CUDA kernel) has two properties
that make it ideal for **constant memory**:

1. **Small**: a 7×7 kernel = 49 floats = 196 bytes. The entire kernel fits in
   constant memory (64 KB on T4), loaded once per CUDA kernel launch.

2. **Read-only and broadcast**: every thread reads the same kernel values in
   the same order. The constant memory cache is optimised for exactly this
   pattern — one fetch broadcasts to all threads in a warp simultaneously.

Declaration:
```cuda
__constant__ float d_kernel[MAX_K * MAX_K];
// copy before launch:
cudaMemcpyToSymbol(d_kernel, h_kernel, K*K*sizeof(float));
```

Inside the kernel, `d_kernel[i * K + j]` hits the constant cache with zero
serialisation — every thread in the warp gets the value in one cycle.

This is **different from shared memory**: constant memory is global (visible
to all blocks) and cached, but read-only. Shared memory is per-block, writable,
and requires explicit load by the threads.

For convolution, the right strategy is:
- Kernel weights  → constant memory
- Input tile      → shared memory (requires collaborative loading + halos)

### Concept 3: Shared Memory Tiling — The Halo Problem

This is the central challenge of tiled 2D convolution and does not exist in
tiled matmul.

In matmul, a TILE_SIZE×TILE_SIZE block of threads loads exactly a
TILE_SIZE×TILE_SIZE tile of A and B. The tile boundaries line up perfectly
with thread boundaries.

In convolution, a block of TILE×TILE output threads needs an input region of
size (TILE + K - 1) × (TILE + K - 1). The extra ring of width K/2 on all
four sides is called the **halo** (or apron or ghost cells).

```
Input tile loaded into shared memory (TILE=4, K=3, halo=1):

  ┌───────────────────────┐
  │  h  h  h  h  h  h    │  ← halo row (top)
  │  h [o][o][o][o] h    │
  │  h [o][o][o][o] h    │  o = output threads own these
  │  h [o][o][o][o] h    │  h = halo cells (loaded but not written)
  │  h [o][o][o][o] h    │
  │  h  h  h  h  h  h    │  ← halo row (bottom)
  └───────────────────────┘
    ^                  ^
    halo col           halo col

  Shared memory size: (TILE + K - 1)² floats
  Output tile size:    TILE²  pixels
```

Consequences:
1. **More threads must load than compute**: with TILE×TILE threads but
   (TILE+K-1)² shared memory cells to fill, some threads must load more than
   one input cell (specifically, halo cells). This requires careful index math.

2. **Halo cells at image boundary may be out-of-bounds**: a thread loading
   a halo cell must check whether its global input index is within [0,H)×[0,W)
   and write 0 into shared memory if it is outside — this is the zero-padding.

3. **__syncthreads() is required after loading**: all TILE² threads must finish
   loading their portion of the shared tile before any thread begins the
   convolution computation.

The halo loading pattern is the most error-prone part of this kernel.
Getting it right is the primary challenge of Version 4.

### Concept 4: Memory Coalescing in 2D

In the naive kernel, each thread reads K×K scattered input pixels:

```
Thread (row, col) reads: Input[row+i][col+j] for i,j in [0,K)
```

Consider a warp of 32 threads (same row, adjacent cols 0..31). At kernel
step (i=0, j=0), thread t reads `Input[row][col+t]` — these are consecutive
→ **coalesced**. Good.

At step (i=0, j=1), thread t reads `Input[row][col+t+1]` — still consecutive
→ **coalesced**. Good.

At step (i=1, j=0), thread t reads `Input[row+1][col+t]` — consecutive in the
next row → **coalesced** (assuming row-major storage). Good.

Unlike matmul's column-access problem in B, convolution's naive kernel is
**already well-coalesced** along the column dimension. The problem is not
coalescing — it is **redundant DRAM fetches** (the same input pixel fetched
K² times across different output threads). Shared memory tiling eliminates
these redundant fetches.

### Concept 5: Tile Size vs Shared Memory Occupancy

Shared memory per block = (TILE + K - 1)² × 4 bytes.

```
K=3, TILE=16: (16+2)² × 4 = 18² × 4 = 1296 bytes  ≈ 1.3 KB
K=3, TILE=32: (32+2)² × 4 = 34² × 4 = 4624 bytes  ≈ 4.5 KB
K=7, TILE=16: (16+6)² × 4 = 22² × 4 = 1936 bytes  ≈ 1.9 KB
K=7, TILE=32: (32+6)² × 4 = 38² × 4 = 5776 bytes  ≈ 5.6 KB
```

T4 shared memory per SM: 64 KB. Even the largest tile above uses under 6 KB,
so shared memory is not the occupancy bottleneck here. Thread count per block
is more likely to limit occupancy (TILE=32 → 1024 threads = max per block on T4).

Optimal tile size: TILE=16 (256 threads, comfortable occupancy) vs
TILE=32 (1024 threads, maximum threads per block). We test both.

### Concept 6: Reference — torch.nn.functional.conv2d (cuDNN)

PyTorch's `F.conv2d` dispatches to **cuDNN** at runtime. cuDNN selects the
fastest algorithm for the given image and kernel size:

- For small kernels (3×3, 5×5): **Winograd algorithm** — reduces FLOPs by
  transforming into a domain where multiplications are cheaper.
- For larger kernels: **FFT-based convolution** or **im2col + cuBLAS**.
- As fallback: direct convolution (similar to our tiled kernel).

cuDNN is the performance ceiling we are trying to approach. Comparing our
kernel time against `F.conv2d` time directly quantifies the gap left by
not implementing Winograd or FFT transforms.

Correctness check usage:
```python
import torch, torch.nn.functional as F

# Our kernel output (numpy array) → torch tensor
our_out = torch.tensor(our_output_numpy)

# Reference
ref_out = F.conv2d(
    input.unsqueeze(0).unsqueeze(0),   # shape: [1, 1, H, W]
    kernel.unsqueeze(0).unsqueeze(0),  # shape: [1, 1, Kh, Kw]
    padding=0
).squeeze()

assert torch.allclose(our_out, ref_out, atol=1e-4), "Mismatch!"
```

The `atol=1e-4` tolerance accounts for float32 accumulation order differences.

---

## Implementation Notes

Kernels are written in CUDA C in `conv_kernels.cu` and compiled from a
Google Colab notebook using `torch.utils.cpp_extension.load` (nvcc, T4 GPU).
The Python harness calls the compiled functions directly, times them with
`torch.cuda.Event`, and compares against `F.conv2d` (cuDNN).

No CPU baseline was implemented. The single-threaded Python loop would be
orders of magnitude slower than any GPU version and is not a useful reference
for optimisation decisions. The meaningful baseline is the naive GPU kernel
itself, and the ceiling is PyTorch/cuDNN.

---

## Versions to Implement

### Version 1: GPU Naive (Global Memory Only)
One thread per output pixel. Each thread reads its K×K input patch directly
from global memory. Kernel weights read from global memory each time.

No shared memory, no constant memory.
Purpose: raw GPU throughput without any memory optimisation.

This version will fetch the same input pixel K² times from DRAM across
different threads — the core inefficiency we then fix.

### Version 2: GPU with Constant Memory for Kernel
Same kernel structure as Version 1, but kernel weights are stored in
`__constant__` memory and loaded once via `cudaMemcpyToSymbol` before launch.

The inner loop `d_kernel[i*K+j]` now hits the broadcast constant cache
instead of L2/DRAM on every thread. All 32 threads in a warp receive the
same kernel weight in one cycle — zero serialisation.

Purpose: isolate the benefit of constant memory for the small, read-only
convolution kernel. Expect a measurable speedup over Version 1 for large K.

### Version 3: GPU Tiled (Shared Memory + Constant Memory)
Thread blocks collaboratively load input tiles (including halos) into shared
memory. All convolution arithmetic is then done from shared memory.
Kernel weights remain in constant memory.

Each input pixel loaded from DRAM exactly once per block — redundant DRAM
fetches eliminated.

Two sub-versions:
- **Version 3a**: TILE_SIZE = 16  (256 threads per block)
- **Version 3b**: TILE_SIZE = 32  (1024 threads per block)

Purpose: primary optimisation demonstration. Quantify the shared-memory
speedup and find the optimal tile size on T4.

### Version 4: torch.nn.functional.conv2d (cuDNN Reference)
The hardware-optimised library baseline. Run with `torch.cuda.Event` timing
(same methodology as our custom kernels).

Purpose: quantify the gap between our best hand-written kernel and cuDNN.
The gap exists because cuDNN uses Winograd / FFT transforms; our kernel uses
direct convolution.

---

## Input Sizes to Test

We sweep two dimensions independently: image size N and kernel size K.

**Image size sweep** (fixed K=3):

| N | Image | Input bytes | Output bytes |
|---|-------|-------------|--------------|
| 512 | 512×512 | 1.0 MB | ~1.0 MB |
| 1024 | 1024×1024 | 4.0 MB | ~4.0 MB |
| 2048 | 2048×2048 | 16.0 MB | ~16.0 MB |
| 4096 | 4096×4096 | 64.0 MB | ~64.0 MB |

**Kernel size sweep** (fixed N=1024):

| K | Kernel | FLOPs per pixel | Theoretical intensity (tiled) |
|---|--------|-----------------|-------------------------------|
| 3 | 3×3 | 18 | 2.25 FLOP/byte |
| 5 | 5×5 | 50 | 6.25 FLOP/byte |
| 7 | 7×7 | 98 | 12.25 FLOP/byte |

This reveals whether the benefit of tiling grows with K (it should, because
larger K means more reuse per input pixel loaded into shared memory).

---

## Performance Metrics

**Primary metric: effective memory bandwidth (GB/s)**

Since all versions are memory-bound for K ≤ 7:

```
bytes_moved = (H × W  +  Kh × Kw  +  out_H × out_W) × 4
bandwidth   = bytes_moved / kernel_time_seconds
```

This is the "minimum bytes" bandwidth — it counts each input and output pixel
once regardless of how many DRAM fetches the kernel actually makes.
A higher number means the kernel is wasting fewer fetches on redundant reads.

**Secondary metric: kernel time (ms)**

Absolute time for fixed N=1024, K=3 across all versions.

**Tertiary: % of T4 peak bandwidth (320 GB/s)**

How close does the tiled kernel get to saturating DRAM bandwidth?

---

## Timing Methodology

Use `torch.cuda.Event` for GPU timing (same methodology as exp3):

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

# warmup (1 run, discarded)
run_kernel(...)

# timed runs
start.record()
for _ in range(NUM_RUNS):
    run_kernel(...)
end.record()
torch.cuda.synchronize()

ms_per_run = start.elapsed_time(end) / NUM_RUNS
```

For Colab T4, use NUM_RUNS = 20 for stable timing.

---

## Hypothesis

### Q1: How much faster is GPU naive vs CPU baseline?
CPU baseline (Python loops) is O(N²K²) with interpreter overhead.
GPU naive launches N² threads in parallel.

Prediction: GPU naive is 100–1000× faster than the Python CPU loop even
without any memory optimisation. The comparison is less meaningful here than
in exp1/2 — Python loops are not the right CPU baseline for production code.
The interesting comparison is Version 2 vs Version 3 vs Version 4 vs Version 5.

### Q2: Does constant memory for the kernel make a measurable difference?
The kernel is tiny (K×K floats ≤ 196 bytes for K=7). It already fits in L2
cache after the first few warps access it in Version 2.

Prediction: constant memory speedup over Version 2 is small for large images
(L2 cache absorbs the kernel anyway) but visible for K=7 and N=512 (where
the kernel-to-output ratio is higher). Do not expect a dramatic speedup here.

### Q3: How much does shared memory tiling improve over naive?
Naive fetches each input pixel K² times from DRAM (through L2, but still
congesting the memory subsystem).
Tiled fetches each input pixel once per block from DRAM.

Prediction: ~2–5× speedup for K=3 (small reuse benefit), ~5–10× for K=7
(larger reuse benefit per input pixel loaded). The tiled kernel's bandwidth
should approach 60–80% of T4 peak (320 GB/s) for large images.

### Q4: TILE_SIZE=16 vs TILE_SIZE=32 — which wins for convolution?
Unlike matmul, the shared memory footprint of tiled convolution is small even
at TILE=32 (≤ 6 KB). So shared memory is not the constraint.

TILE=32 means 1024 threads per block (max on T4). Higher thread count per
block → better latency hiding but also higher register pressure.
TILE=16 means 256 threads per block → more blocks per SM → better wave
utilisation for images that are not huge multiples of 32.

Prediction: TILE=32 slightly faster for large N (better reuse ratio),
TILE=16 slightly better for N=512 (more blocks → better SM utilisation).
Difference likely < 20% — not as dramatic as matmul's tiling tradeoff.

### Q5: How close does our tiled kernel get to cuDNN?
cuDNN uses Winograd for K=3 and K=5, which reduces FLOPs by 2.25× and 2×
respectively while keeping the same output correctness. Our kernel does direct
convolution (no Winograd).

Prediction: our tiled kernel achieves 30–60% of cuDNN throughput for K=3
(Winograd gap is significant here). For K=7 (cuDNN falls back to direct),
our kernel gets much closer — possibly within 10–20% of cuDNN.

### Q6: Does the tiled kernel's speedup over naive grow with K?
Arithmetic reuse per input pixel = K². More reuse → more benefit from tiling.

Prediction: yes, speedup of tiled over naive grows monotonically with K.
At K=3 the speedup might be 2–4×. At K=7 it should be 6–10×.

---

## Forward — Experiment 5 (Planned)

The remaining gap between our tiled kernel and cuDNN at large N, K=3 comes
from the **Winograd F(2,3) algorithm**, which reduces multiplications from
9 to 4 per output element (2.25× fewer FLOPs). Implementing Winograd from
scratch is deferred to **Experiment 5**, after completing the relevant PMPP
chapters on algorithmic optimisations.

---

## What Success Looks Like

After this experiment I should be able to:
    ✅ Implement 2D thread indexing for image kernels confidently
    ✅ Explain why convolution is memory-bound for small kernels (K ≤ 7)
    ✅ Use __constant__ memory correctly and explain when it helps vs L2 cache
    ✅ Implement the halo-loading pattern for tiled convolution without bugs
    ✅ Calculate the shared memory tile size including halo: (TILE + K - 1)²
    ✅ Explain why coalescing is NOT the main problem in convolution (unlike matmul)
    ✅ Measure effective bandwidth and compare to T4 peak
    ✅ Quantify the gap between our kernel and cuDNN / explain why it exists (Winograd)
    ✅ Connect this experiment to CNN inference: conv layers are 60–70% of compute
       in models like ResNet, so understanding this kernel = understanding why
       GPU inference hardware (tensor cores, im2col) is designed the way it is
