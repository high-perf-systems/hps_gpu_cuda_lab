# Experiment 6: Parallel Histogram

## Problem Statement

Given an input array `data` of N integers, each already bucketed into
`[0, num_bins)`, compute a histogram `H` where:

```
H[b] = count of i such that data[i] == b,   for b = 0 .. num_bins-1
```

This is the simplest possible form of the PMPP "parallel histogram" problem
(pre-binned integer categories, not raw values needing a bin-width division) —
the wrinkle is not the arithmetic, it's *where each thread writes*.

---

## Connection to Previous Experiments

| Experiment | Operation | Output location per thread | Memory pattern | Bottleneck |
|------------|-----------|------------------------------|-----------------|------------|
| Exp 1: Vec add | C[i] = A[i]+B[i] | fixed, unique per thread | 1 read → 1 write | Bandwidth |
| Exp 2: Matmul | C[r][c] = dot(row,col) | fixed, unique per thread | input reused N times | Compute |
| Exp 3: Reduction | result = Σ A[i] | fixed, **shared** by all threads, via a known tree schedule | tree of writes | Bandwidth |
| Exp 4/5: Conv/Stencil | O[r][c] = Σ patch | fixed, unique per thread | input reused K/6r+1 times | Bandwidth |
| **Exp 6: Histogram** | **H[data[i]]++** | **shared, and *data-dependent*** | **scatter, unknown write pattern until runtime** | **Contention / serialization, not bandwidth** |

Reduction (exp3) was the first time multiple threads had to combine into the
same output — but the *pattern* of who-combines-with-whom was fixed and known
at compile time (a binary tree), which is exactly what let us design around
it with `__syncthreads()` and shuffles. Histogram breaks that assumption:
which threads collide on which output location depends entirely on the
**input data**, not on thread ID. Two threads processing adjacent array
elements might write to the same bin, or to opposite ends of the histogram —
you cannot know until you look at the data. This is why histogram needs a
different tool: **atomic read-modify-write operations**, which let any thread
safely update any location without a pre-planned schedule, at the cost of
serializing whichever threads happen to collide at runtime.

---

## Core Concepts

### Concept 1: Atomic Operations — GPU vs CPU

An atomic operation (`atomicAdd(&H[bin], 1)`) performs a read-modify-write as
one indivisible step: no other thread can observe or interleave with it
partway through. Without atomics, two threads incrementing the same bin
concurrently can race (both read the old value, both write old+1 — one
increment is lost).

This is conceptually identical to CPU atomics (`std::atomic<int>::fetch_add`,
or a locked `xadd` instruction) — same contract, same underlying idea of
serializing conflicting read-modify-writes via the memory system's coherence
protocol. The difference is **scale of contention**: on a CPU, at most a
handful of cores can ever collide on one cache line at once. On a GPU,
thousands of threads can issue an atomic to the *same address* in the same
cycle. The hardware queues and serializes them, so the cost of an atomic
scales with how many concurrent threads target that address — not a fixed
cost the way it effectively is on a CPU. This is the central bottleneck this
experiment is built around, and the motivation for every technique below.

### Concept 2: Contention Is a Property of the Data, Not Just the Kernel

Because the write address is `data[i]`-dependent, the *same kernel* can be
fast or slow purely depending on the input distribution:

- **Uniform random data** spreads writes evenly across bins → low contention,
  atomics rarely collide.
- **Skewed data** (most values landing in a few hot bins) → many threads
  hammer the same address → heavy serialization, regardless of how well the
  kernel is otherwise written.

This makes data distribution a first-class experimental variable here, in
the same way N was the swept variable in exp3.

### Concept 3: Privatization

Instead of every thread contending for the same global histogram, each
**thread block** keeps a private copy of the histogram in shared memory.
Threads atomically update their block's private copy (shared-memory atomics
are much faster than global-memory atomics, and contention is now bounded by
one block's worth of threads instead of the whole grid). At the end, each
block merges its private histogram into the global one with `num_bins`
atomic adds per block — far fewer global atomics than one per input element.

### Concept 4: Thread Coarsening and Partitioning Strategy

Having each thread process multiple input elements reduces per-element
overhead, but *how* the elements are assigned to threads matters:
- **Contiguous partitioning** — thread *t* handles a contiguous chunk
  `[t*k, t*k+k)`. Simple, but adjacent threads read far-apart memory —
  memory accesses are not coalesced.
- **Interleaved partitioning** — thread *t* handles elements
  `t, t+stride, t+2*stride, ...`. Adjacent threads read adjacent memory on
  each step → coalesced loads, at the cost of a slightly less obvious
  indexing scheme.

### Concept 5: Aggregation

If consecutive elements processed by a thread often fall in the *same* bin
(common in sorted or locally-clustered data), a thread can accumulate a
running count for "the bin I'm currently matching" in a register and issue
one `atomicAdd(&H[bin], run_length)` instead of one atomic per element —
collapsing a run of atomics into a single one. This only pays off when runs
actually exist in the data; for uniform random data, consecutive elements
almost never share a bin, so aggregation should show little or no benefit
there.

---

## Reference, Correctness, and Performance-Floor Strategy

Two comparison points, each with a distinct role (mirroring exp3's dual
CPU-baseline + thrust::reduce structure, adapted to this experiment):

- **`numpy.bincount(data, minlength=num_bins)`** as the CPU reference. This is
  the direct CPU equivalent of exp3's `std::accumulate` baseline — a single-
  threaded, library-implemented, sequential routine, not multi-threaded and
  not GPU-accelerated. Same role there: exp3 didn't hand-write a reduction
  loop either, it called into an existing sequential library primitive. This
  is the **correctness oracle** every GPU version is checked against, and it
  also gives an honest single-threaded performance floor.
- **`cub::DeviceHistogram::HistogramEven`** (CUB ships with the CUDA toolkit,
  so it's usable directly from the `.cu` file compiled by
  `torch.utils.cpp_extension.load` — no separate build system needed). This
  is the **optimized-library performance ceiling**, the same role
  `thrust::reduce` played in exp3: how close does our best hand-written
  kernel get to NVIDIA's own tuned implementation?

`torch.bincount` was considered and set aside as redundant with CUB for this
role — CUB is the more direct "vendor-optimized C++ implementation" analog
used elsewhere in this lab, and keeping everything in the `.cu` file avoids
mixing two different reference implementations.

---

## Build Order (Intentionally Incremental)

Unlike exp3-5, where every kernel version was fully specified in `problem.md`
before writing any code, this experiment starts narrower on purpose — the
right shape for privatization/coarsening/aggregation will be easier to judge
after seeing the naive kernel's actual contention behavior on real data:

1. **CPU reference** — `numpy.bincount`, correctness oracle + floor. Wire this
   up first and sanity-check it against all planned data distributions.
2. **GPU Version 1: Naive (global-memory atomics)** — one thread per input
   element, `atomicAdd(&H[data[i]], 1)` directly into a global-memory
   histogram. No shared memory, no privatization. This is the only GPU
   kernel specified right now.
3. **Later, added incrementally**: privatized (Concept 3), coarsened with a
   partitioning choice (Concept 4), and aggregated (Concept 5) versions —
   each designed after observing where V1 actually loses time. Document the
   actual progression and any detours in `notes.md` as they happen, the way
   exp5 documents why V4 (rolling shared planes) was tried before landing on
   V5 (register tiling).
4. **Final step**: compare the best hand-written version against
   `cub::DeviceHistogram::HistogramEven`.

---

## Data to Generate

All arrays are `int32`, N elements, values pre-binned into `[0, num_bins)`.

**Distributions** (the key independent variable — see Concept 2):
- **Uniform** — `torch.randint(0, num_bins, (N,))`. Low-contention case.
- **Skewed** — most mass concentrated in a small fraction of bins (e.g. a
  clipped Gaussian centered on one bin, or a Zipfian draw). High-contention
  case; this is what privatization is expected to fix.

**Bin counts**: `num_bins ∈ {8, 32, 256}` — fewer bins forces more threads to
share each bin even under a uniform distribution, so this sweep isolates
"contention from bin granularity" from "contention from data skew."

**N sweep**: `{1M, 4M, 16M, 64M}` elements, consistent with exp3's reduction
sweep scale.

---

## Performance Metrics

Histogram is contention-bound, not bandwidth- or compute-bound in the usual
roofline sense (output is tiny and fixed-size regardless of N) — so the
metrics center on throughput and on quantifying the contention penalty
directly, rather than FLOP/byte roofline analysis:

- **Primary: throughput** — `N / time` (elements/sec).
- **Secondary: input bandwidth** — `N × 4 bytes / time` (GB/s), reported
  mainly to show how far below peak bandwidth a contention-bound kernel sits
  even though it's "just" streaming through the input once.
- **Contention penalty (the key number for this experiment)**:
  `time(skewed) / time(uniform)`, same kernel, same N and num_bins. Tracked
  across every version to show which techniques close this gap and by how
  much.

---

## Timing Methodology

Identical to exp4/exp5 — `torch.cuda.Event`, 3 warmup runs discarded, 20
timed runs averaged.

---

## Hypotheses

Only stated for what's actually being built now (V1 + CPU reference); later
versions get their own predictions in `notes.md` once their design is fixed,
per the incremental build order above.

### Q1: How much does data skew degrade the naive kernel?
Prediction: large — skewed data concentrates atomics on a few bins, and with
potentially millions of threads issuing atomics grid-wide, the naive kernel
has no mechanism to bound contention. Expect an order-of-magnitude (5-20×)
slowdown for skewed vs uniform data at the same N.

### Q2: How does num_bins affect the naive kernel, even under uniform data?
Prediction: fewer bins measurably slows the naive kernel even with uniform
data, since more threads necessarily collide per bin purely from pigeonhole
counting (N / num_bins expected collisions per bin). Expect this effect to be
much smaller than the skew effect in Q1.

### Q3: How does the naive GPU kernel compare to the CPU reference?
Prediction: GPU wins even at its worst (skewed, few bins) for large N, since
even heavily serialized global atomics still run at GPU clock speed across
many contention queues in parallel across bins — but the margin should shrink
sharply from the uniform case to the skewed case, unlike every prior
experiment where the GPU/CPU gap was roughly stable across data variants.

---

## Forward — What Comes After V1

- Privatized version, expected to mostly close the uniform/skewed gap from
  Q1 (contention now bounded per-block, not per-grid).
- Coarsened version with an interleaved partitioning choice, expected to
  reduce merge overhead from privatization without sacrificing coalescing.
- Aggregated version, expected to help specifically on data with local runs
  (not the uniform/skewed distributions above — likely needs a third,
  sorted/run-length-heavy distribution to actually exercise this).
- Final comparison against `cub::DeviceHistogram::HistogramEven`.

---

## What Success Looks Like

After this experiment I should be able to:
    ✅ Explain why GPU atomic contention is a bigger practical problem than CPU atomic contention, and why (thread count sharing an address)
    ✅ Explain why histogram's bottleneck is contention/serialization, not bandwidth or compute — and why the roofline model from exp2-5 doesn't directly apply here
    ✅ Demonstrate, with real measurements, that the SAME kernel's performance depends on the input DATA distribution, not just its code
    ✅ Implement and correctness-verify a naive global-atomic histogram kernel against a hand-written CPU reference
    ✅ Explain privatization, coarsening/partitioning, and aggregation conceptually, ready to implement and measure each incrementally
    ✅ Explain why cub::DeviceHistogram is used as a performance ceiling here (parallel: thrust::reduce in exp3), not a correctness shortcut
