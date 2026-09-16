# Experiment 6: Parallel Histogram — Benchmark Results and Analysis (V1-V4)

## Implementation Summary

All four kernels live in a single CUDA C file, compiled from a Google Colab
notebook via `torch.utils.cpp_extension.load`, targeting a T4 GPU (Compute
Capability 7.5). The Python harness calls each compiled kernel directly,
times it with `torch.cuda.Event`, and checks correctness against
`numpy.bincount` — which also serves as the CPU performance floor
(`time.perf_counter`, transfer excluded from the timed region).

**Files:**
- `src/histogram.cu` — `launch_naive`, `launch_privatized`, `launch_coarsening`,
  `launch_aggregation` + pybind11 wrapper
- `src/histogram_reference.py` — data generation (`generate_uniform`,
  `generate_skewed`), CPU reference (`cpu_histogram`, `cpu_bincount_timer`),
  correctness check (`verify`)
- `src/histogram_harness.py` — compiles the kernel, kernel registry,
  correctness sweep, benchmark sweep, all result tables below

---

## Setup

| Item | Value |
|------|-------|
| GPU | NVIDIA T4 (Google Colab) |
| Compute Capability | 7.5 (Turing) |
| T4 peak DRAM bandwidth | 320 GB/s |
| CUDA version | 12.x |
| PyTorch version | 2.x |
| Warmup runs | 3 (discarded) |
| Timed runs | 20 (averaged) |
| GPU timer | `torch.cuda.Event` |
| CPU timer | `time.perf_counter` (H2D-independent — `.cpu()` transfer excluded from timed region) |
| Input dtype | int32, pre-binned into `[0, num_bins)` |
| Threads/block | 256 (all versions) |
| Grid (naive, privatized) | `ceil(N/256)` blocks, boundary-guarded (`if (idx < N)`) |
| Grid (coarsened, aggregation) | `ceil(N/2048)` blocks (`C_FACTOR=8`), grid-stride loop over `stride = blockDim.x * gridDim.x` |
| N sweep | 1M, 4M, 16M, 64M |
| num_bins sweep | 8, 32, 256 |
| Distributions | uniform (`torch.randint`), skewed (`torch.multinomial`, 1 hot bin carrying 90% of mass) |

---

## Kernel Versions

| # | Name (harness key) | Description |
|---|------|-------------|
| 1 | `1_naive` | One thread per input element. `atomicAdd(&histogram[data[idx]], 1)` directly into global memory — no shared memory, no privatization. |
| 2 | `2_privatized` | Concept 3: each block keeps a private histogram in shared memory (`extern __shared__ int histo_s[]`, sized `num_bins * 4` bytes). Threads atomically update the block-local copy, then `num_bins` atomics per block merge it into the global histogram. Bounds contention to one block's worth of threads instead of the whole grid. |
| 3 | `3_coarsened` | Concept 4: same shared-memory privatization as V2, but each block processes `C_FACTOR=8` elements per thread via an **interleaved** grid-stride loop (`for (i=idx; i<N; i+=blockDim.x*gridDim.x)`) — fewer blocks, more work per block, coalesced loads preserved since adjacent threads still read adjacent addresses on each step. |
| 4 | `4_aggregation` | Concept 5: identical shape to V3, but each thread keeps a register-level `(prevIdx, accumulator)` pair across its grid-stride loop. Consecutive visits to the *same* bin are merged into one `atomicAdd(&histo_s[bin], run_length)` instead of one atomic per element; the running accumulator is flushed on a bin change and once more after the loop. |

---

## Results

All timings: 3 warmup + 20 timed runs, one shared random dataset per
(N, num_bins, distribution) cell. This is a single consolidated sweep run
after V4 was added, so V1-V3 numbers here supersede the V1-only sweep from
the previous revision of this file (differences are normal Colab run-to-run
variance, a few percent at most — see Finding 5 for a case where that
variance actually matters).

### Kernel Time (ms)

**num_bins=8 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 0.4775 | 1.8768 | 4.4863 | 21.2482 |
| 2_privatized | 0.0460 | 0.1711 | 0.3170 | 1.1982 |
| 3_coarsened | 0.0360 | 0.1141 | 0.2533 | 0.9815 |
| 4_aggregation | 0.0372 | 0.1253 | 0.2623 | 1.0509 |

**num_bins=8 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 1.3281 | 5.2857 | 15.8262 | 49.4262 |
| 2_privatized | 0.0999 | 0.3726 | 0.6279 | 2.3951 |
| 3_coarsened | 0.0957 | 0.3370 | 0.5153 | 1.9573 |
| 4_aggregation | 0.0416 | 0.1402 | 0.2691 | 1.0477 |

**num_bins=32 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 0.4971 | 1.9501 | 4.3841 | 20.8148 |
| 2_privatized | 0.0424 | 0.1466 | 0.2801 | 1.1079 |
| 3_coarsened | 0.0359 | 0.1002 | 0.2450 | 0.9655 |
| 4_aggregation | 0.0346 | 0.1109 | 0.2535 | 1.0317 |

**num_bins=32 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 1.3896 | 5.5287 | 16.4498 | 51.4106 |
| 2_privatized | 0.0986 | 0.2663 | 0.6037 | 2.3941 |
| 3_coarsened | 0.0940 | 0.2358 | 0.4918 | 1.9546 |
| 4_aggregation | 0.0407 | 0.1079 | 0.2637 | 1.0449 |

**num_bins=256 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 0.5835 | 2.3030 | 9.1740 | 24.5160 |
| 2_privatized | 0.1005 | 0.3767 | 1.0769 | 3.3348 |
| 3_coarsened | 0.0344 | 0.0895 | 0.2517 | 0.9615 |
| 4_aggregation | 0.0382 | 0.0948 | 0.2600 | 1.0165 |

**num_bins=256 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 1.4029 | 5.5847 | 17.2644 | 51.1086 |
| 2_privatized | 0.1116 | 0.2996 | 0.7167 | 2.6698 |
| 3_coarsened | 0.0985 | 0.2087 | 0.5427 | 2.0271 |
| 4_aggregation | 0.0445 | 0.0776 | 0.2786 | 1.0711 |

### Effective Bandwidth (GB/s) — T4 peak = 320 GB/s

**num_bins=8 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 8.38 | 8.53 | 14.27 | 12.05 |
| 2_privatized | 87.01 | 93.52 | 201.88 | 213.66 |
| 3_coarsened | 111.07 | 140.27 | 252.67 | 260.83 |
| 4_aggregation | 107.47 | 127.67 | 243.95 | 243.59 |

**num_bins=8 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 3.01 | 3.03 | 4.04 | 5.18 |
| 2_privatized | 40.02 | 42.94 | 101.92 | 106.89 |
| 3_coarsened | 41.81 | 47.48 | 124.21 | 130.79 |
| 4_aggregation | 96.12 | 114.11 | 237.86 | 244.35 |

**num_bins=32 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 8.05 | 8.20 | 14.60 | 12.30 |
| 2_privatized | 94.28 | 109.11 | 228.50 | 231.08 |
| 3_coarsened | 111.51 | 159.66 | 261.25 | 265.14 |
| 4_aggregation | 115.77 | 144.32 | 252.42 | 248.13 |

**num_bins=32 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 2.88 | 2.89 | 3.89 | 4.98 |
| 2_privatized | 40.59 | 60.08 | 106.01 | 106.93 |
| 3_coarsened | 42.55 | 67.86 | 130.13 | 130.97 |
| 4_aggregation | 98.36 | 148.23 | 242.73 | 245.00 |

**num_bins=256 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 6.86 | 6.95 | 6.98 | 10.44 |
| 2_privatized | 39.78 | 42.48 | 59.43 | 76.77 |
| 3_coarsened | 116.30 | 178.73 | 254.30 | 266.26 |
| 4_aggregation | 104.73 | 168.75 | 246.20 | 251.83 |

**num_bins=256 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 2.85 | 2.86 | 3.71 | 5.01 |
| 2_privatized | 35.85 | 53.41 | 89.30 | 95.89 |
| 3_coarsened | 40.60 | 76.67 | 117.92 | 126.29 |
| 4_aggregation | 89.83 | 206.16 | 229.70 | 239.00 |

### Speedup vs CPU (`np.bincount`)

**num_bins=8 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 6.21× | 8.41× | 14.06× | 11.83× |
| 2_privatized | 64.50× | 92.22× | 198.94× | 209.86× |
| 3_coarsened | 82.33× | 138.33× | 249.00× | 256.19× |
| 4_aggregation | 79.66× | 125.90× | 240.40× | 239.26× |

**num_bins=8 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 2.63× | 2.59× | 4.77× | 6.50× |
| 2_privatized | 34.93× | 36.74× | 120.29× | 134.20× |
| 3_coarsened | 36.48× | 40.63× | 146.59× | 164.22× |
| 4_aggregation | 83.89× | 97.64× | 280.73× | 306.79× |

**num_bins=32 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 5.54× | 5.29× | 14.48× | 12.26× |
| 2_privatized | 64.91× | 70.41× | 226.64× | 230.27× |
| 3_coarsened | 76.78× | 103.03× | 259.12× | 264.21× |
| 4_aggregation | 79.71× | 93.14× | 250.36× | 247.26× |

**num_bins=32 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 2.54× | 2.56× | 4.57× | 5.84× |
| 2_privatized | 35.77× | 53.08× | 124.51× | 125.42× |
| 3_coarsened | 37.51× | 59.94× | 152.84× | 153.62× |
| 4_aggregation | 86.69× | 130.94× | 285.09× | 287.36× |

**num_bins=256 — uniform**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 5.25× | 4.53× | 8.60× | 11.38× |
| 2_privatized | 30.44× | 27.68× | 73.25× | 83.67× |
| 3_coarsened | 88.98× | 116.48× | 313.45× | 290.22× |
| 4_aggregation | 80.13× | 109.98× | 303.45× | 274.49× |

**num_bins=256 — skewed**

| version | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|
| 1_naive | 2.47× | 2.48× | 4.66× | 5.88× |
| 2_privatized | 31.07× | 46.17× | 112.30× | 112.48× |
| 3_coarsened | 35.19× | 66.28× | 148.29× | 148.14× |
| 4_aggregation | 77.85× | 178.21× | 288.85× | 280.34× |

### Contention Penalty = time(skewed) / time(uniform)

| version | num_bins | N=1M | N=4M | N=16M | N=64M |
|---|---|---|---|---|---|
| 1_naive | 8 | 2.78 | 2.82 | 3.53 | 2.33 |
| 1_naive | 32 | 2.80 | 2.84 | 3.75 | 2.47 |
| 1_naive | 256 | 2.40 | 2.43 | 1.88 | 2.08 |
| 2_privatized | 8 | 2.17 | 2.18 | 1.98 | 2.00 |
| 2_privatized | 32 | 2.32 | 1.82 | 2.16 | 2.16 |
| 2_privatized | 256 | 1.11 | 0.80 | 0.67 | 0.80 |
| 3_coarsened | 8 | 2.66 | 2.95 | 2.03 | 1.99 |
| 3_coarsened | 32 | 2.62 | 2.35 | 2.01 | 2.02 |
| 3_coarsened | 256 | 2.86 | 2.33 | 2.16 | 2.11 |
| 4_aggregation | 8 | 1.12 | 1.12 | 1.03 | 1.00 |
| 4_aggregation | 32 | 1.18 | 0.97 | 1.04 | 1.01 |
| 4_aggregation | 256 | 1.17 | 0.82 | 1.07 | 1.05 |

---

## Key Findings

### Finding 1 — The first skewed-data generator confounded skew severity with `num_bins`, and had to be fixed mid-experiment

The initial `generate_skewed(N, num_bins, hot_fraction=0.1, ...)` computed
`num_hot = round(num_bins * hot_fraction)` — an absolute hot-bin count that
scales *with* `num_bins`, which entangled "contention from bin granularity"
with "contention from data skew" (the two things `problem.md`'s `num_bins`
sweep was meant to isolate). Fixed by making the hot-bin count an absolute
parameter (`num_hot=1`, `hot_mass=0.9`, decoupled from `num_bins`), so 90% of
mass lands in exactly one bin regardless of `num_bins`. All results in this
file use the corrected generator.

**Takeaway:** the bug was in the experimental methodology, not the CUDA
kernel — worth double-checking data-generation code as carefully as kernel
code when a result looks "too clean."

### Finding 2 — Every version beats the CPU at every configuration, and privatization is where the real win happens

Naive (V1) only manages 2.5×-14× over `np.bincount`. Privatization (V2)
jumps that to 27×-230× — a bigger single-version jump than any of the
subsequent refinements. Coarsening (V3) and aggregation (V4) push the
ceiling further (up to ~313× and ~307× respectively), but the qualitative
shift from "modest CPU win" to "two-orders-of-magnitude CPU win" happens at
the naive → privatized boundary, confirming `problem.md` Concept 3's framing:
bounding atomic contention to one block is the single highest-leverage
change in this experiment.

### Finding 3 — Privatization (V2) shrinks the naive kernel's raw contention penalty from ~2-4× to ~0.7-2.3×, and the shrinkage is largest at high bin counts

V1's contention penalty ranges 1.88-3.75× across all configs. V2's ranges
0.67-2.32× — better everywhere, but the improvement is not uniform: at
`num_bins=8/32` the penalty only drops to ~1.8-2.3× (privatization still
leaves real block-level contention on the single hot bin), while at
`num_bins=256` it collapses to 0.67-1.11× — skewed data runs *as fast or
faster* than uniform. This only partially confirms the informal prediction
in the previous revision of this file ("expected to mostly close the
uniform/skewed gap") — it fully closes at high bin count but only partially
at low bin count.

Mechanism (traced during V2/V3 development, not yet confirmed with `ncu`):
the merge phase's `if (bin_val > 0) atomicAdd(...)` skips the global atomic
entirely for any shared bin a block never touched. With `num_hot=1` fixed
skew, each 256-thread block's ~10% "cold" mass (26 elements on average at
`num_bins=8`, spread thinner as `num_bins` grows) leaves most of the 255
cold bins empty at `num_bins=256` (~90% empty, Poisson-thin), so far fewer
merge-phase global atomics fire under skew than under the uniform case
(~37% empty at that bin count) — enough to outweigh the real accumulate-
phase contention cost and flip the penalty below 1.0. At `num_bins=8/32`
there aren't enough cold bins for this skip-benefit to matter, so the
underlying accumulate-phase contention dominates and the penalty stays
above 1.0.

### Finding 4 — Coarsening (V3) gives a further, consistent ~10-70% raw speedup over V2, largest at high bin count

Comparing V2 → V3 kernel time directly (uniform, N=64M): bins=8 goes
1.198→0.982ms (18% faster), bins=32 goes 1.108→0.966ms (13% faster),
bins=256 goes 3.335→0.961ms (**71% faster**). The `num_bins=256` case is the
standout: V2's per-block shared histogram there is 256 ints (1KB) that must
be zero-initialized and merged by every one of the many small V2 blocks,
so cutting the block count 8× (via `C_FACTOR=8`) cuts that fixed per-block
overhead 8×, which matters far more when `num_bins` — and therefore that
fixed cost — is large. This confirms the informal V3 prediction ("reduce
merge overhead from privatization without sacrificing coalescing") cleanly,
and explains why V3's `num_bins=256` bandwidth (254-266 GB/s) is dramatically
higher than V2's (39-77 GB/s) even though both use the same privatization
strategy.

Under skewed data the V2→V3 win is real but much flatter across `num_bins`
(~1.22-1.32× at N=64M for bins=8/32/256, vs. uniform's 1.02×-3.47× spread) —
coarsening packs the same total hot-bin atomic count into 8× fewer, longer
serialized per-block queues (less cross-block latency-hiding), and each
block now also absorbs ~8× more cold elements, which thins out V2's
merge-skip benefit (the cold-bin-empty fraction at `num_bins=256` drops from
~90% under V2's smaller blocks to ~45% under V3's — see Finding 3's
mechanism). That's also why V3's `num_bins=256` contention penalty
(2.11-2.86×) sits *above* 1.0 again, undoing V2's below-1.0 anomaly at that
bin count.

### Finding 5 — Aggregation (V4) is the first version to nearly close the uniform/skewed gap outright, and does so via a fundamentally different mechanism than V2/V3

This directly confirms what you flagged from the logs. Isolating the
V3 → V4 kernel-time ratio (coarsened time / aggregation time; >1 means
aggregation is faster) across every `(num_bins, N)` cell:

| distribution | bins=8 | bins=32 | bins=256 |
|---|---|---|---|
| uniform | 0.91×-0.97× (**3-9% slower**) | 0.90×-1.04× (mostly 4-10% slower; N=1M is a wash) | 0.90×-0.97× (**3-10% slower**) |
| skewed | 1.87×-2.40× faster | 1.87×-2.31× faster | 1.89×-2.69× faster |

- **Skewed:** aggregation is faster than coarsening at *every single*
  `(num_bins, N)` combination tested (12/12), by 1.87×-2.69×. This drives
  V4's contention penalty down to **0.82-1.18×** across the board — the only
  version where skewed data doesn't run measurably slower than uniform data,
  a qualitatively different result from V1-V3's persistent 1.9-3.8× penalty.
- **Uniform:** aggregation is slower than coarsening at 11 of 12 cells (the
  one exception, bins=32/N=1M, is a 3.8% difference well inside normal
  run-to-run noise). The regression is small (3-10%) but consistent, exactly
  as you predicted: with uniform data there is close to nothing to
  aggregate — under `num_bins=8`, the run-length is only ~1.14 elements on
  average (`1/(1-1/8)`), so almost every loop iteration pays the
  `(num == prevIdx)` branch and the extra `accumulator`/`prevIdx` register
  traffic for essentially zero atomic-count reduction. It's pure overhead
  with nothing to amortize it against.

**Why this works mechanically (and a caveat on `problem.md` Concept 5):**
both V3 and V4 use the same interleaved grid-stride loop, with
`stride = blockDim.x * gridDim.x` — roughly `N/8` for `C_FACTOR=8`. That
means the handful of elements (~8) a single thread visits are **not**
spatially adjacent in `data[]`; they are ~N/8 apart. So the "runs" V4's
accumulator collapses are not genuine spatial/local runs in the sense
Concept 5 originally describes (sorted or locally-clustered data) — the
generators here (`generate_uniform`, `generate_skewed`) are pure i.i.d.
categorical draws with zero spatial structure. What V4 actually exploits is
simpler: for skewed data, any two draws (however far apart in the array)
land in the same hot bin with probability ≈0.9 each, so a thread's own
*strided* sequence still contains long same-bin runs purely from marginal
probability (mean run length `1/(1-p) ≈ 10` at `hot_mass=0.9`), and that's
enough to trigger the accumulator. This is a **more general** trigger than
Concept 5's original framing anticipated — the previous revision of this
file predicted aggregation would show "little or no benefit" on the
uniform/skewed distributions and would need a dedicated sorted/local-run
distribution to be exercised at all. That prediction was too narrow: skew
alone, even without any sorting, is sufficient. A genuinely sorted/
run-length-heavy third distribution (still on the Forward list) would now
mainly serve to test whether *contiguous* partitioning (thread *t* handles
`[t*k, t*k+k)`) beats the current interleaved partitioning specifically
for aggregation — since contiguous partitioning is the only way this
kernel would ever see true spatially-adjacent runs.

### Finding 6 — N=16M's elevated-penalty anomaly (previous revision, V1 only) did not reproduce cleanly in this run

The prior version of this file flagged N=16M as having an unexplained,
consistently elevated naive contention penalty across all three `num_bins`
values. In this sweep, N=16M is still elevated for bins=8 (3.53×) and
bins=32 (3.75×), but *not* for bins=256 (1.88×, actually the lowest of the
four N values at that bin count). Since the same anomaly no longer holds
"consistently across all three `num_bins`" in a second independent run,
this weakens the case for a real N=16M-specific hardware effect and
strengthens the "likely Colab GPU-sharing noise" reading — still flagged
rather than dismissed outright, and still worth an isolated repeat run
before either explaining or discarding it (see Forward).

---

## Hypothesis Review

### V1 (from `problem.md`)

| Hypothesis | Prediction | Actual | Verdict |
|---|---|---|---|
| Q1: Skew degrades naive kernel by 5-20× | large (order of magnitude) | ~1.9-3.8× at true worst-case (single hot bin) skew | ❌ Wrong magnitude — direction correct, but GPU atomic throughput per address appears bounded-but-fast rather than serializing toward CPU-like cost |
| Q2: Fewer bins measurably slow the naive kernel even under uniform data (pigeonhole collisions) | small negative effect | No clear monotonic trend; spread <25%, likely within noise | ⚠️ Inconclusive — needs `ncu` profiling, not confirmed either direction |
| Q3: GPU beats CPU even at its worst, but margin shrinks sharply under skew | yes | Confirmed — GPU wins everywhere (2.5×-14× at its worst), margin narrows sharply under skew at every V1 config | ✅ Confirmed |

### V2-V4 (informal predictions from the previous revision's "Forward" section)

| Version | Prediction | Actual | Verdict |
|---|---|---|---|
| V2 Privatized | "expected to mostly close the uniform/skewed gap from Q1" | Closes fully at `num_bins=256` (penalty 0.67-1.11×), only partially at `num_bins=8/32` (penalty 1.82-2.32×) | ⚠️ Partially confirmed — bin-count-dependent |
| V3 Coarsened | "expected to reduce merge overhead from privatization without sacrificing coalescing" | Confirmed — 10-70% faster than V2, with the largest gain (71%) exactly where per-block merge overhead is largest (`num_bins=256`) | ✅ Confirmed |
| V4 Aggregation | "expected to help specifically on data with local runs... likely needs a third sorted distribution to actually exercise this" | Confirmed for skewed (1.87-2.69× over V3) and for uniform (small, consistent 3-10% regression) — but the *mechanism* prediction was too narrow: skew alone (no sorting/locality needed) is sufficient to trigger the benefit under the current interleaved partitioning, since it only needs marginal-probability repeats, not spatial adjacency (Finding 5) | ✅ Confirmed on outcome, ❌ mechanism assumption revised |

---

## What Was Confirmed

- ✅ GPU beats the naive CPU `np.bincount` baseline at every tested configuration and every version
- ✅ Privatization (V2) is the single highest-leverage change in the whole experiment — it, not coarsening or aggregation, produces the jump from "double-digit× over CPU" to "two-orders-of-magnitude× over CPU"
- ✅ Coarsening (V3) delivers a further, real (10-70%) speedup over privatization alone, with gains scaling with `num_bins` (more shared memory to init/merge per block → more to save by having fewer, bigger blocks)
- ✅ Aggregation (V4) is the only version that nearly eliminates the uniform/skewed performance gap (penalty ~0.8-1.2× vs. 1.9-3.8× for V1 and 1.8-2.3×/2.0-2.9× for V2/V3), by collapsing repeated-bin atomics into one — but at a small, consistent cost (3-10% slower) on uniform data where there's nothing to collapse

## What Was Surprising

- ❌ The original skewed-data generator accidentally coupled skew severity to `num_bins` (Finding 1) — a methodology bug, not a kernel bug
- ❌ Contention penalty for the naive kernel topped out around ~3.8×, far below Q1's predicted 5-20×
- ❌ Aggregation's benefit on skewed data did **not** require spatial locality or a sorted distribution, contradicting the previous revision's stated expectation — the interleaved grid-stride loop scatters a thread's ~8 visited elements roughly N/8 apart, so the "runs" being collapsed are purely a marginal-probability effect of skew, not genuine data locality (Finding 5)
- ⚠️ N=16M's elevated naive-kernel contention penalty (flagged in the prior V1-only run) did not reproduce at `num_bins=256` in this independent run, weakening the case for a real hardware effect there (Finding 6)

---

## Forward — What Comes Next

1. **`ncu` profiling** — `--section SpeedOfLight --section WarpStateStats --section MemoryWorkloadAnalysis` across V1-V4, both distributions. Specifically needed to confirm (rather than infer):
   - Finding 3's `num_bins=256` mechanism for why V2's uniform/skewed gap closes so much more than at `num_bins=8/32`.
   - Q2's still-inconclusive `num_bins` effect on the naive kernel under uniform data.
2. **Repeat N=16M in isolation** for the naive kernel specifically at `num_bins=256`, now that Finding 6 shows the "elevated N=16M penalty" pattern didn't hold there in a second run — settle whether this is noise (increasingly likely) or a real, narrower effect.
3. **A genuinely sorted / locally-clustered third distribution**, now specifically to test **contiguous vs. interleaved partitioning for aggregation** (Finding 5's caveat) — the current interleaved kernel already benefits from skew alone, so this distribution's job is to isolate whether true spatial locality plus contiguous partitioning would help V4 further, not to "unlock" aggregation for the first time.
4. **Final comparison against `cub::DeviceHistogram::HistogramEven`**, per `problem.md`'s original build order — the last remaining item before this experiment is complete.
