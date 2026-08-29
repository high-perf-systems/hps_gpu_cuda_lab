# Experiment 5: 3D Stencil — Benchmark Results and Analysis

## Implementation Summary

Kernels were written in CUDA C and compiled from a Google Colab notebook via
`torch.utils.cpp_extension.load`, targeting a T4 GPU (Compute Capability 7.5).
The Python harness (embedded in the notebook) called the compiled CUDA
functions directly, timed them with `torch.cuda.Event`, and checked
correctness against `F.conv3d` with a dense, zero-padded kernel built from
the coefficient tables in `problem.md`.

No CPU baseline was implemented, for the same reason as exp4: a Python/NumPy
triple-nested-loop reference would be orders of magnitude slower and is not
a useful baseline for optimization decisions.

**Files:**
- `stencil_kernels.cu` — all five CUDA kernel variants + pybind11 wrappers
- `stencil.ipynb` — correctness checks, benchmark harness, results (self-contained; the earlier standalone `stencil_reference.py` / `stencil_harness.py` were superseded once the notebook became self-contained and are kept only as historical scaffolding)

---

## Setup

| Item | Value |
|------|-------|
| GPU | NVIDIA T4 (Google Colab) |
| Compute Capability | 7.5 (Turing) |
| T4 peak DRAM bandwidth | 320 GB/s |
| T4 peak FP32 compute | 8,141 GFLOPS |
| T4 ridge point | 25.4 FLOP/byte |
| CUDA version | 12.x |
| PyTorch version | 2.x |
| Warmup runs | 3 (discarded) |
| Timed runs | 20 (averaged) |
| Timer | `torch.cuda.Event` |
| Input dtype | float32 |
| Grid shape | cubic N×N×N |
| Boundary handling | valid mode, no padding |
| Output shape | (N−2r) × (N−2r) × (N−2r) |
| Radius sweep | 1, 2, 3 (2nd/4th/6th order) |

---

## Kernel Versions

| # | Name (harness key) | Description |
|---|------|-------------|
| 1 | `1_naive` | One thread per output point. Reads its `6r+1` neighbors directly from global memory; coefficients passed as a kernel argument. |
| 2 | `2_constmem` | Same structure as naive, but coefficients live in `__constant__` memory (`cudaMemcpyToSymbol`), broadcast to a warp in one cycle instead of a per-thread read. |
| 3 | `3_shared_tiled` | Full 3D shared-memory tile: block loads a `(8+2r)³` haloed input cube into shared memory, `__syncthreads()`, then computes. `in_tile` is sized exactly to the runtime `radius` (dynamic shared memory), not the compile-time max radius — no over-fetch. |
| 4 | `4_thread_coarse` | Thread coarsening, first attempt: a 2D (x,y) block sweeps a chunk of z-depths in a loop, keeping a rolling window of `2r+1` **fully haloed** x-y planes in shared memory (drop oldest, load one new leading plane, per z-step). Grid is chunked in z (`gridDim.z = ceil(oD/16)`) to avoid the occupancy collapse a single-block-per-column design causes. |
| 5 | `4_register_tiled`* | Register tiling: same coarsened/chunked block structure as V4, but only the **center** x-y plane is haloed and kept in shared memory. The `2r` z-neighbor taps never need an x/y halo (the stencil is an axis-aligned cross — a z-tap always reads the same (row,col) as the center), so they're kept in a small per-thread register array instead, templated on `radius` so the tap loop fully unrolls and the array stays in registers rather than spilling to local memory. |

\* The harness key is `"4_register_tiled"`, but this is chronologically and conceptually **Version 5** — V4 (thread coarsening) was built and benchmarked first, before its underperformance (see Finding 3/4 below) motivated V5. `problem.md`'s "Versions to Implement" section reflects this actual build order; the harness dict key is a naming leftover from before the split.

---

## Arithmetic Intensity

Recap from `problem.md` (unchanged by these results — AI is a property of
the algorithm, not the implementation):

| Radius | Order | Tiled AI (FLOP/byte) | Naive AI (FLOP/byte) |
|--------|-------|----------------------|------------------------|
| 1 | 2nd | 1.75 | ≈0.5 |
| 2 | 4th | 3.25 | ≈0.5 |
| 3 | 6th | 4.75 | ≈0.5 |

All values sit far below the T4's 25.4 FLOP/byte ridge point — this
operation was never going to be compute-bound at any radius tested. Every
finding below is about *how much wasted DRAM traffic* each kernel design
avoids, not about extracting more FLOPs.

---

## Results

All timings: 3 warmup + 20 timed runs, `torch.cuda.Event`, one shared random
input per (N, radius) cell across all five versions.

### Kernel Time (ms)

**radius=1**

| version | N=64 | N=128 | N=256 | N=384 |
|---|---|---|---|---|
| 1_naive | 0.0825 | 0.1680 | 1.5665 | 5.7523 |
| 2_constmem | 0.0360 | 0.1657 | 1.5553 | 5.5350 |
| 3_shared_tiled | 0.1296 | 0.2641 | 2.6464 | 8.6498 |
| 4_thread_coarse | 0.0777 | 0.2576 | 2.3153 | 8.3585 |
| 4_register_tiled | 0.0360 | 0.1789 | 1.6084 | 5.7757 |

**radius=2**

| version | N=64 | N=128 | N=256 | N=384 |
|---|---|---|---|---|
| 1_naive | 0.0362 | 0.2184 | 2.1216 | 7.8296 |
| 2_constmem | 0.0351 | 0.2130 | 2.1458 | 7.7313 |
| 3_shared_tiled | 0.0341 | 0.3007 | 3.0163 | 9.9999 |
| 4_thread_coarse | 0.0585 | 0.3833 | 3.2678 | 11.4785 |
| 4_register_tiled | 0.0410 | 0.2049 | 1.7458 | 6.2472 |

**radius=3**

| version | N=64 | N=128 | N=256 | N=384 |
|---|---|---|---|---|
| 1_naive | 0.0360 | 0.3070 | 2.8819 | 10.3742 |
| 2_constmem | 0.0377 | 0.3136 | 2.8347 | 9.8927 |
| 3_shared_tiled | 0.0374 | 0.3910 | 3.5881 | 12.1449 |
| 4_thread_coarse | 0.0786 | 0.6183 | 4.9149 | 17.4913 |
| 4_register_tiled | 0.0433 | 0.2297 | 1.9504 | 6.8187 |

### Effective Bandwidth (GB/s) — T4 peak = 320 GB/s

**radius=1**

| version | N=64 | N=128 | N=256 | N=384 |
|---|---|---|---|---|
| 1_naive | 24.27 | 97.53 | 84.68 | 78.14 |
| 2_constmem | 55.68 | 98.94 | 85.30 | 81.20 |
| 3_shared_tiled | 15.45 | 62.06 | 50.13 | 51.96 |
| 4_thread_coarse | 25.76 | 63.62 | 57.29 | 53.77 |
| 4_register_tiled | 55.65 | 91.62 | 82.48 | 77.82 |

**radius=2**

| version | N=64 | N=128 | N=256 | N=384 |
|---|---|---|---|---|
| 1_naive | 52.89 | 73.32 | 61.80 | 56.96 |
| 2_constmem | 54.45 | 75.19 | 61.10 | 57.69 |
| 3_shared_tiled | 56.09 | 53.25 | 43.47 | 44.60 |
| 4_thread_coarse | 32.72 | 41.78 | 40.13 | 38.85 |
| 4_register_tiled | 46.59 | 78.17 | 75.10 | 71.39 |

**radius=3**

| version | N=64 | N=128 | N=256 | N=384 |
|---|---|---|---|---|
| 1_naive | 50.75 | 50.98 | 44.97 | 42.66 |
| 2_constmem | 48.52 | 49.91 | 45.72 | 44.73 |
| 3_shared_tiled | 48.96 | 40.03 | 36.12 | 36.44 |
| 4_thread_coarse | 23.26 | 25.31 | 26.37 | 25.30 |
| 4_register_tiled | 42.19 | 68.15 | 66.45 | 64.90 |

### GFLOPS (T4 FP32 peak = 8,141)

| radius | version | N=64 | N=128 | N=256 | N=384 |
|---|---|---|---|---|---|
| 1 | 1_naive | 40.5 | 166.7 | 146.5 | 135.7 |
| 1 | 2_constmem | 92.8 | 169.1 | 147.5 | 141.0 |
| 1 | 3_shared_tiled | 25.8 | 106.0 | 86.7 | 90.2 |
| 1 | 4_thread_coarse | 42.9 | 108.7 | 99.1 | 93.4 |
| 1 | 4_register_tiled | 92.8 | 156.5 | 142.6 | 135.1 |
| 2 | 1_naive | 155.3 | 226.9 | 196.1 | 182.2 |
| 2 | 2_constmem | 159.9 | 232.7 | 193.9 | 184.5 |
| 2 | 3_shared_tiled | 164.7 | 164.8 | 137.9 | 142.7 |
| 2 | 4_thread_coarse | 96.1 | 129.3 | 127.3 | 124.3 |
| 2 | 4_register_tiled | 136.8 | 242.0 | 238.3 | 228.4 |
| 3 | 1_naive | 205.7 | 224.7 | 206.0 | 197.8 |
| 3 | 2_constmem | 196.7 | 220.0 | 209.5 | 207.5 |
| 3 | 3_shared_tiled | 198.5 | 176.5 | 165.5 | 169.0 |
| 3 | 4_thread_coarse | 94.3 | 111.6 | 120.8 | 117.3 |
| 3 | 4_register_tiled | 171.0 | 300.5 | 304.4 | 301.0 |

Best-ever GFLOPS achieved (register-tiled, radius=3, N=256) is **304.4** —
3.7% of the T4's 8,141 GFLOPS peak. Confirms the arithmetic-intensity
prediction: this experiment was never going to approach the compute
roofline at any radius; GFLOPS is a diagnostic, not a target metric here.

### Speedup vs Naive (bandwidth ratio, N=384)

| version | radius=1 | radius=2 | radius=3 |
|---|---|---|---|
| 2_constmem | 1.04× | 1.01× | 1.05× |
| 3_shared_tiled | 0.66× | 0.78× | 0.85× |
| 4_thread_coarse | 0.69× | 0.68× | 0.59× |
| 4_register_tiled | 1.00× | 1.25× | 1.52× |

---

## Key Findings

### Finding 1 — Shared-memory tiling (V3) never beats naive, at any radius tested

```
N=384, effective bandwidth (GB/s):
  radius=1:  naive 78.1   vs  shared_tiled 51.96   (0.66×)
  radius=2:  naive 56.96  vs  shared_tiled 44.60   (0.78×)
  radius=3:  naive 42.66  vs  shared_tiled 36.44   (0.85×)
```

This directly contradicts `problem.md`'s Q2 prediction of a "2–6× speedup,
growing with radius." Two effects compound against V3, reasoned through
during development rather than confirmed with a profiler (no `ncu` run was
collected this round — see Forward):

1. **T4's 4 MB L2 cache is already doing V3's job for free.** Per-block
   working sets here top out in the tens of KB — tiny relative to L2 — so
   the "redundant" per-thread reads naive/const-mem issue are very likely
   L2 cache hits, not real DRAM round-trips. Shared-memory tiling is paying
   an explicit synchronization + load-loop cost to manually re-derive reuse
   the hardware was already giving away.
2. **`__syncthreads()` is a hard, unconditional stall** every 512-thread
   block pays once per launch — naive/const-mem have zero cross-thread
   waits, since every thread is fully independent.

Interestingly the *ratio* does move in the predicted direction (0.66× →
0.78× → 0.85×, closing toward parity as radius grows) — reuse potential
really does grow with radius, as Q2 argued — it just never closes far
enough to cross 1.0× within the radii tested here.

### Finding 2 — Naive thread coarsening (V4) is the worst version, and gets worse with radius

```
N=384 bandwidth (GB/s), radius=1 → 2 → 3:
  4_thread_coarse:  53.77 → 38.85 → 25.30   (steadily falling)
  1_naive:          78.14 → 56.96 → 42.66   (also falling, but far less steeply)
```

V4's ratio-to-naive falls from 0.69× (r=1) to 0.59× (r=3) — the opposite of
what coarsening was supposed to buy. Root cause, worked out while debugging
this version directly: V4's rolling z-window keeps **`2·radius+1` fully
haloed x-y planes** in shared memory, but only the *center* plane's halo is
ever read — the z-neighbor taps always read the exact same (row, col) as
the center thread, never an offset one, because the stencil is an
axis-aligned cross, not a dense cube. At radius=3 with a 16×16 output tile
(`in_tile=22`), each side plane's useful "core" is `16²=256` of its `22²=484`
loaded cells — **47% of every side plane's shared-memory traffic is
provably dead weight**, and there are 6 side planes doing this per rolling
step. The waste scales with radius (more side planes), which is exactly why
V4's relative performance degrades as radius increases while every other
version's relative performance holds steady or improves.

(V4's grid also originally collapsed to a handful of blocks — one block per
x-y tile summing the *entire* z-depth in a loop — which independently
starved the GPU of parallelism; chunking the z-sweep into `gridDim.z =
ceil(oD/16)` blocks fixed that specific problem, but the halo-waste issue
above remained and turned out to be the larger, harder-to-fix cost.)

### Finding 3 — Register tiling (V5) is the only version that beats naive, and its lead grows with radius

```
N=384, speedup vs naive (bandwidth ratio):
  radius=1: 1.00×   radius=2: 1.25×   radius=3: 1.52×
```

This is the cleanest confirmatory result in the experiment, and it inverts
V3/V4's trend exactly as `problem.md`'s Concept 3 predicted: eliminating
V4's wasted side-plane halos (Finding 2) by keeping only the center plane
haloed in shared memory, and moving the `2·radius` z-taps into a
per-thread register array (templated on `radius` so the compiler can fully
unroll the tap loop and keep that array in real registers instead of
spilling to local memory), removes almost exactly the waste that made V4
degrade with radius — so V5's advantage over naive *grows* with radius
instead of shrinking. At radius=1 there's only one side plane on each side
of center, so there was never much halo waste to eliminate, and V5 ends up
essentially tied with naive/const-mem (77.8 vs 78.1 GB/s at N=384) rather
than ahead of them — consistent with the mechanism, not a separate anomaly.

V5 also beats V3 (plain 3D shared tiling) at every radius tested, and that
gap *also* grows with radius (V5/V3 bandwidth ratio at N=384: 1.50× at r=1,
1.60× at r=2, 1.78× at r=3) — confirming `problem.md` Q4's prediction that
register tiling should pull further ahead of full 3D tiling as radius
increases, though contrary to Q4's specific hedge, V5 does *not* lose to V3
at radius=1 — it wins there too, just by a smaller margin.

### Finding 4 — Constant memory gives a small, radius-independent edge, mostly visible at small N

```
N=128, bandwidth (GB/s):
  radius=1: naive 97.5  vs constmem 98.9   (1.01×)
  radius=2: naive 73.3  vs constmem 75.2   (1.03×)
  radius=3: naive 51.0  vs constmem 49.9   (0.98×, within noise)
```

Matches `problem.md` Q1's prediction of a small effect (< 10%, here
1–5% at meaningful sizes) — the coefficient array is tiny (2–4 floats),
so the broadcast-cache benefit that mattered so much in exp4 (weights read
K² times per output pixel) barely registers here, since coefficients are
read only `radius+1` times per output point and the dominant cost is the
`6r+1` *input* reads, which const-mem doesn't touch. At N=64 the gap looks
much larger (up to 2.3× at radius=1) — see Finding 5 for why N=64 numbers
in general shouldn't be read too literally.

### Finding 5 — N=64 timings are noisy; don't over-read them

```
radius=1, N=64: naive 0.0825 ms vs radius=2, N=64: naive 0.0362 ms
                (fewer taps at r=2 should not run in less time than r=1)
```

At N=64 every kernel's total runtime is 30–130 microseconds — the same
order of magnitude as fixed CUDA kernel-launch overhead (typically
5–20 μs) and within range of ordinary clock-boost/scheduling variance
between consecutive kernel launches. The radius=1/N=64 naive result above
is the clearest symptom: a stencil with fewer taps runs *slower* than the
same kernel with more taps at the same N, which has no algorithmic
explanation and is best read as launch-overhead noise, not a real
signal. N≥128 results, where runtimes are ≥0.16 ms, are far more trustworthy
for comparing kernel designs; all "Key Findings" above are stated using
N=384 or trends that hold consistently from N=128 up.

### Finding 6 — Even the best kernel stays far from the bandwidth roofline

```
Best raw bandwidth achieved anywhere: 98.9 GB/s (constmem, N=128, r=1)
                                        = 30.9% of T4's 320 GB/s peak
```

Comparable to exp4's convolution ceiling (97.2 GB/s / 30.4% of peak,
constmem at K=3) — this stencil's raw GB/s ceiling turned out *not* to be
dramatically lower than convolution's, contrary to a naive reading of Q6.
What *is* worse, exactly as Q6's arithmetic-intensity reasoning predicted,
is useful FLOPs extracted per byte moved: at radius=3 (19-point stencil,
6·3+1=19 taps) the best GFLOPS reached is 304 versus exp4's K=7
(49-tap dense) kernels reaching comparable-or-better bandwidth while doing
far more FLOPs per byte (12.1 vs 4.75 FLOP/byte, per each experiment's own
arithmetic-intensity table). The two operations hit similar bandwidth
ceilings on the same hardware; the stencil just extracts less useful work
per byte while doing so, because its 6r+1-point cross is inherently sparser
than a same-radius dense cube.

---

## Hypothesis Review

| Hypothesis | Prediction | Actual | Verdict |
|---|---|---|---|
| Q1: Constant memory gives small (<10%) benefit | small | 1–5% at N≥128 (N=64 noisier, up to 2.3×) | ✅ Correct at meaningful sizes |
| Q2: Shared tiling (V3) gives 2–6× speedup, growing with radius | large speedup | **Never beats naive** at any radius tested (0.66×–0.85×) | ❌ Wrong direction — but the *ratio* does close toward parity as radius grows, as predicted |
| Q3: V3 hits an occupancy wall at radius=3/large tiles | yes | Not directly measured (no `ncu` occupancy profiling collected) | ⚠️ Not verified this round |
| Q4: Register tiling (V5) beats V3, margin grows with radius; V3 competitive at r=1 | V5 wins at large r, V3 competitive at r=1 | V5 beats V3 at **every** radius (1.50×–1.78×), margin does grow with radius | ✅ Core claim confirmed, ❌ "V3 competitive at r=1" hedge was wrong — V5 already wins there |
| Q5: Tiled-vs-naive speedup grows with radius | yes | False for V3 (plateaus below 1.0×); **true and clean for V5** (1.00×→1.25×→1.52×) | ✅ True once V5 is the "tiled" kernel meant |
| Q6: Worse bandwidth *per useful FLOP* than exp4's conv2D | yes | Confirmed — similar raw GB/s ceiling to exp4, but far fewer FLOPs/byte extracted at matched radius/K | ✅ Confirmed, with the nuance that raw GB/s ceilings ended up similar, not lower |

---

## What Was Confirmed

- ✅ Constant memory gives a small, radius-insensitive edge (Q1)
- ✅ Register tiling (V5) is the only kernel that beats naive, and its lead over both naive and full 3D tiling grows with radius (Q4, Q5)
- ✅ This stencil is deeply memory-bound at every radius tested — GFLOPS never exceeds 3.7% of peak compute
- ✅ Stencils extract far fewer useful FLOPs per byte than a same-radius dense convolution, even at a comparable raw bandwidth ceiling (Q6)
- ✅ The axis-aligned "cross" sparsity of a stencil (vs. a dense cube) is exactly what register tiling exploits — z-taps need no x/y halo at all

## What Was Surprising

- ❌ Full 3D shared-memory tiling (V3) never beat naive/const-mem at any tested radius — the predicted "large" tiling win (Q2) simply didn't materialize; L2 cache absorption appears to dominate at these working-set sizes
- ❌ The first, more literal reading of "thread coarsening + register tiling" (V4: coarsen with fully-haloed rolling planes, no register offload) was actively *worse* than naive, and got worse with radius — the opposite of what coarsening was supposed to buy. The register-only offload for z-taps (V5) turned out to be the load-bearing idea, not the coarsening/chunking by itself
- ❌ Register tiling (V5) beat V3 even at radius=1, where `problem.md` predicted the two might be competitive
- ⚠️ N=64 measurements were noisy enough to show an algorithmically-impossible result (radius=1 slower than radius=2 for the same kernel) — a reminder to sanity-check the smallest benchmark size before trusting it

---

## Ranking of Optimizations by Radius (N=384 bandwidth ratio vs naive)

```
radius=1:  register_tiled (1.00×) ≈ naive (1.00×) ≈ constmem (1.04×) > thread_coarse (0.69×) > shared_tiled (0.66×)
radius=2:  register_tiled (1.25×) > constmem (1.01×) ≈ naive (1.00×) > shared_tiled (0.78×) > thread_coarse (0.68×)
radius=3:  register_tiled (1.52×) > constmem (1.05×) ≈ naive (1.00×) > shared_tiled (0.85×) > thread_coarse (0.59×)
```

Register tiling is the only version whose ranking *improves* with radius;
every other version either stays flat (naive, const-mem) or gets
progressively worse (shared-tiled, thread-coarse) as radius grows.

---

## What Success Looks Like — Achieved

- ✅ Implemented a 3D axis-aligned stencil kernel (naive) confidently
- ✅ Explained why a stencil is more memory-bound than a same-radius dense convolution (Finding 6)
- ✅ Calculated 3D shared-memory tile cost including halo, and confirmed empirically that it doesn't pay off here (Finding 1)
- ✅ Implemented register-tiled z-sweep stencil, including debugging two real correctness bugs along the way (a `__syncthreads()` barrier-divergence crash from an early `return` in the thread-coarsened version, and an input/output-space indexing mix-up in the register array) before reaching a passing, benchmarkable kernel
- ✅ Explained the register-vs-shared-memory tradeoff for 3D neighborhoods, and *why* it specifically favors the z-axis for this cross-shaped stencil (Finding 3)
- ✅ Measured effective bandwidth and compared across stencil orders (radius 1/2/3) for all five kernel versions
- ✅ Explained why `torch.conv3d`/cuDNN is a correctness oracle here, not a performance ceiling, and confirmed why via the GFLOPS/AI comparison in Finding 6

---

## Forward — Possible Experiment 6

`problem.md`'s original forward-looking items remain open, plus two new ones
this round's results motivate directly:

1. **`ncu` profiling of V3 and V4** to directly confirm (rather than infer)
   the L2-cache-absorption and barrier-stall explanations behind Findings 1
   and 2 — `--section SpeedOfLight --section WarpStateStats --section
   MemoryWorkloadAnalysis`, per this repo's CC-7.5 profiling notes. In
   particular, checking `stall_sync`/`stall_barrier` on V3 and V4 would
   confirm or rule out the barrier-stall half of Finding 1's explanation.
2. **Sweep `Z_CHUNK` and `OUT_TILE_TC` independently** for V5 — both are
   currently tied together (`Z_CHUNK = OUT_TILE_TC = 16`); decoupling and
   sweeping them separately (as noted while building V5) may push register
   tiling's already-leading numbers further.
3. Thread coarsening tuned per-radius (originally planned for this
   experiment, effectively superseded once V5's register-based design beat
   coarsening-alone outright — worth revisiting only if (1) or (2) reopen
   the question).
4. Multi-GPU halo exchange — splitting a large 3D grid across GPUs and
   exchanging boundary planes — remains untouched and is the most open item
   carried forward from `problem.md`.
