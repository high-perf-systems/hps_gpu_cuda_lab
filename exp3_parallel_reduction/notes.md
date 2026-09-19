# Notes: Experiment 3 — Parallel Reduction

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
| **Warp Size** | 32 threads |
| **Max Threads per Block** | 1024 |
| **Driver / CUDA Version** | 580.82.07 / 12.8 |
| **Build Flags** | `nvcc -O2` |
| **Threads per Block** | 256 (all versions) |
| **Warmup Runs** | 3 (discarded) |
| **Timed Runs** | 10 (averaged) |

---

## 1. Problem Framing

Reduction maps N elements to 1 scalar via an associative operation:

```
result = A[0] + A[1] + ... + A[N-1]
```

Unlike exp1 (vector add) and exp2 (matmul), reduction is not embarrassingly
parallel. Every thread must eventually contribute to the same output value,
requiring coordination. The two core challenges are:

1. **Synchronisation** — threads writing to shared locations must coordinate
   to avoid data races.
2. **Degrading parallelism** — after each step, half the threads become idle.
   The GPU's parallelism advantage shrinks as the computation converges.

---

## 2. Arithmetic Intensity and Expected Bottleneck

```
Operations per element  :  O(log₂ N) additions  ≈  20 adds for N=1M
Bytes read per element  :  4 bytes (one float, read essentially once)
Arithmetic intensity    :  ~20 / 4  ≈  5 FLOP/byte
T4 Ridge Point          :  8141 / 320  =  25.4 FLOP/byte
5 << 25.4               →  reduction is MEMORY BOUND by roofline model
```

Theoretical best case at N=16M:
```
64 MB at 320 GB/s  =  0.2 ms  (absolute floor — no kernel can beat this)
```

**Important caveat discovered from ncu:** The roofline model correctly
identifies the ceiling but misidentifies the actual binding bottleneck
for naive implementations. Data is loaded from DRAM once into shared memory,
and all subsequent steps happen on-chip. The real bottleneck for V1 and V2
is synchronisation overhead, not DRAM bandwidth.

---

## 3. Hypotheses

### 3.1 CPU Baseline

Single-threaded sequential sum. IEEE 754 non-associativity prevents
vectorisation without `-ffast-math`. The bottleneck is the serial
accumulator dependency chain — each add must wait for the previous result.

**Predicted CPU time: ~1-2 ms for N=16M.**

### 3.2 Version 1 — Interleaved Addressing

```cpp
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

Step s=1 has threads 0,2,4,6... active and threads 1,3,5... idle per warp.
50% throughput loss. Averages to 12.5% thread utilisation across all 8 steps.

**Predicted bandwidth: 50-100 GB/s (15-30% of peak).**

### 3.3 Version 2 — Sequential Addressing

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

Active threads are always 0..s-1. Whole warps are either fully active or
fully idle. Divergence eliminated for first 3 steps (s >= 32). Residual
divergence remains for last 5 steps (s < 32) within the final warp.

**Predicted speedup over V1: 1.5-2x.**

### 3.4 Version 3 — First Add During Load

**Predicted speedup over V2: 1.3-1.7x.**
Halves sync barriers by doing one add during global load.

**MEASURED: 1.92x at large N (1.18x at N=1M).** Exceeded the predicted range.
The prediction was based on barrier reduction alone (1 fewer of 8). The actual
mechanism turned out to be mostly the improved memory access pattern — two
coalesced loads per thread raised DRAM throughput from 19% to 34%. The barrier
reduction was the minor contributor. Prediction was right in direction, wrong
in mechanism and magnitude.

### 3.5 Version 4 — Unroll Last Warp

**Predicted speedup over V3: 1.1-1.3x.**
Removes 5 `__syncthreads()` calls. Constant improvement regardless of N.

**MEASURED: 1.55x at small N, 2.75x at 64M.** Far exceeded the prediction, and
the prediction's "constant improvement regardless of N" was WRONG — the gain
grows sharply with N. The prediction missed two things: (1) removing barriers
let bandwidth jump at large N as the kernel crossed from sync-bound to
memory-bound (section 7.6), and (2) the unroll introduced an idle-warp tail that
dropped occupancy 89→73% — a cost not anticipated. Right direction, wrong
magnitude, and wrong on the N-scaling.

### 3.6 Version 5 — Warp Shuffle

**Predicted speedup over V4: 1.2-1.5x.**
No shared memory for last warp, no barriers for final 5 steps.

**MEASURED: 1.04x (1M), 1.08x (4M), 1.10x (16M), 1.15x (64M) — median of 4
clean runs.** Undershot the predicted range at every N, and the *shape* was
wrong too. The hypothesis (formed after understanding V4's plateau-then-jump,
§7.6) was: "V5 helps at small N — latency-bound, fewer dependent shared-mem
ops in the tail — and barely moves large N, since V4 is already at the
memory wall with no headroom." Measured is flat-to-slightly-*rising* with
N — the opposite direction. ncu on V6 (§6.9, which shares V5's shuffle tail)
explains why: Warp Cycles Per Issued Instruction stayed at 19.88 —
statistically the same as V4's 18.4 and V3's 20.1 — and the dominant stall
(34.7%, "L1TEX scoreboard dependency," i.e. long-scoreboard) comes from the
global-load / early tree-reduction phase, not the last-warp shared-memory
traffic shuffle replaced. Right direction, small magnitude, wrong shape —
see §7.7 for the full account.

### 3.7 Version 6 — Hierarchical Atomic Reduction

*(Not in the original problem.md — added after reading PMPP's "hierarchical
reduction for arbitrary input length," which combines block results via a
single-kernel `atomicAdd` instead of this file's original two-pass
partial-sums + CPU-combine approach.)*

**Predicted: replacing the per-block write + CPU combine with
`atomicAdd(output, val)` costs measurably more per block (atomic RMW
contention across ~131,072 blocks funnelling into one 4-byte address at
N=64M), but wins on architecture — one kernel launch, no D2H copy, no host
loop, works for arbitrary N.**

**MEASURED: V6/V5 = 0.98-1.03x across N (median of 4 runs) — statistically
indistinguishable from V5, not slower.** The predicted contention cost did
not materialise. Only `40 SMs × 4 resident blocks/SM = 160` blocks are
actually executing at any instant, out of 131,072 total at N=64M, so the
atomicAdd calls are naturally staggered across the kernel's lifetime rather
than genuinely simultaneous. See §7.8 for the full explanation. Prediction
direction was wrong: this turned out to be a free architectural upgrade, not
a performance/simplicity tradeoff.

### 3.8 Thrust Reference

**Predicted: V5 reaches 50-70% of Thrust bandwidth.**

**MEASURED: V5 reaches 97-594% of Thrust across N** — 97% at 16M (Thrust's
best showing, actually ~3% ahead of V5 there), up to 594% at 1M (Thrust's
fixed dispatch overhead dominates at small N), and 122% at 64M. The
prediction assumed V5 would still trail Thrust everywhere; instead V4
already matched/beat Thrust at 64M (§4.6), and V5/V6 extend that lead
further at every N except 16M.

### 3.9 Predicted Summary Table

| Version | Key Mechanism | Predicted BW | % of Peak |
|---------|-------------|-------------|-----------|
| CPU | Serial accumulator | ~50 GB/s | N/A |
| V1 | Warp divergence | 50-100 GB/s | 15-30% |
| V2 | No divergence (first 3 steps) | 100-150 GB/s | 30-47% |
| V3 | Half sync barriers | 130-180 GB/s | 40-56% |
| V4 | No barriers last warp | 150-200 GB/s | 47-62% |
| V5 | Warp shuffle | 180-220 GB/s | 56-69% |
| V6 | Hierarchical atomic combine | ~= V5 (no cost predicted to hold) | ~= V5 |
| Thrust | All optimisations | ~260-280 GB/s | 80-87% |

---

## 4. Results

### 4.1 Timing Results

```
N = 1,048,576  (4.2 MB)
    CPU               :    3.277 ms  (  1.3 GB/s)
    V1 Interleaved    :    0.130 ms  ( 32.4 GB/s, 10.1% peak)  [OK]
    V2 Sequential     :    0.081 ms  ( 51.9 GB/s, 16.2% peak)  [OK]
    Thrust            :    0.176 ms  ( 23.9 GB/s,  7.5% peak)  [OK]

N = 4,194,304  (16.8 MB)
    CPU               :   12.619 ms  (  1.3 GB/s)
    V1 Interleaved    :    0.548 ms  ( 30.6 GB/s,  9.6% peak)  [OK]
    V2 Sequential     :    0.335 ms  ( 50.1 GB/s, 15.7% peak)  [OK]
    Thrust            :    0.217 ms  ( 77.3 GB/s, 24.2% peak)  [OK]

N = 16,777,216  (67.1 MB)
    CPU               :   53.725 ms  (  1.2 GB/s)
    V1 Interleaved    :    2.628 ms  ( 25.5 GB/s,  8.0% peak)  [OK]
    V2 Sequential     :    1.608 ms  ( 41.7 GB/s, 13.0% peak)  [OK]
    Thrust            :    0.604 ms  (111.1 GB/s, 34.7% peak)  [OK]

N = 67,108,864  (268.4 MB)
    CPU               :  212.324 ms  (  1.3 GB/s)
    V1 Interleaved    :   10.464 ms  ( 25.7 GB/s,  8.0% peak)  [OK]
    V2 Sequential     :    6.401 ms  ( 41.9 GB/s, 13.1% peak)  [OK]
    Thrust            :    1.274 ms  (210.8 GB/s, 65.9% peak)  [OK]
```

### 4.2 V2 Speedup Over V1

| N | V2 / V1 Speedup |
|---|----------------|
| 1M | 1.60x |
| 4M | 1.64x |
| 16M | 1.63x |
| 64M | 1.63x |

Consistent ~1.63x across all N. This is the signature of a **fixed per-block
improvement** — the divergence elimination saves a constant fraction of time
per block regardless of how many blocks run. If the gain came from better
memory bandwidth utilisation, it would scale with N like Thrust does.

### 4.3 Bandwidth Progression

| N | V1 GB/s | V2 GB/s | Thrust GB/s |
|---|---------|---------|------------|
| 1M | 32.4 | 51.9 | 23.9 |
| 4M | 30.6 | 50.1 | 77.3 |
| 16M | 25.5 | 41.7 | 111.1 |
| 64M | 25.7 | 41.9 | 210.8 |

Both V1 and V2 show flat bandwidth across all N (~27 and ~42 GB/s respectively).
Thrust scales from 24 GB/s to 210 GB/s. The flat line is the diagnostic
signature of a sync-bound kernel. Bandwidth scales with N only when memory
is the actual bottleneck. V1 and V2 are sync-bound.

### 4.4 Hypothesis Validation

| Prediction | Expected | Measured | Verdict |
|-----------|---------|---------|---------|
| CPU time at N=16M | 1-2 ms | 51 ms | ✗ Serial dependency chain much slower than predicted |
| V1 bandwidth | 50-100 GB/s | 27 GB/s | ✗ Lower — sync overhead more dominant than predicted |
| V2 speedup over V1 | 1.5-2x | 1.63x | ✓ Within range |
| V2 bandwidth | 100-150 GB/s | 42 GB/s | ✗ Still sync-bound, lower than predicted |
| Both V1/V2 flat across N | — | Confirmed | ✓ Both ~flat, Thrust scales |
| GPU beats CPU | Yes | Yes (~20-33x kernel only) | ✓ |

### 4.5 Version 3 Results — First Add During Load

```
N = 1,048,576  (4.2 MB)
    V3 AddLoad       :    0.081 ms  ( 52.0 GB/s, 16.2% peak)  [OK]
N = 4,194,304  (16.8 MB)
    V3 AddLoad       :    0.218 ms  ( 77.1 GB/s, 24.1% peak)  [OK]
N = 16,777,216  (67.1 MB)
    V3 AddLoad       :    0.841 ms  ( 79.8 GB/s, 24.9% peak)  [OK]
N = 67,108,864  (268.4 MB)
    V3 AddLoad       :    1.694 ms  (158.5 GB/s, 49.5% peak)  [OK]
```

**Bandwidth progression now includes V3:**

| N | V1 GB/s | V2 GB/s | V3 GB/s | Thrust GB/s |
|---|---------|---------|---------|------------|
| 1M | 27.3 | 44.1 | 52.0 | 18.5 |
| 4M | 25.1 | 41.0 | 77.1 | 65.0 |
| 16M | 25.6 | 41.6 | 79.8 | 141.6 |
| 64M | 25.7 | 82.6 | 158.5 | 215.5 |

**V3 / V2 incremental speedup:**

| N | V3 / V2 |
|---|---------|
| 1M | 1.18x |
| 4M | 1.88x |
| 16M | 1.92x |
| 64M | 1.92x |

**The key qualitative shift:** V1 bandwidth was dead flat (~26 GB/s at all N) —
the textbook sync-bound signature. V3 bandwidth *rises with N* (52 → 158 GB/s).
This upward slope is the kernel beginning to behave like a memory-bound kernel
rather than a purely sync-bound one. The headline ~1.92x speedup matters less
than this change in shape: the speedup itself now grows with N, which neither
V1 nor V2 achieved at the smaller sizes.

At N=64M, V3 reaches 49.5% of peak bandwidth and 73.5% of Thrust's bandwidth —
up from V1's 12% of Thrust. The gap to Thrust is closing fast.

The V3/V2 ratio at N=1M (1.18x) is smaller than at larger N (1.92x) because at
N=1M the problem is too small to amortise launch overhead — the same small-N
noise seen throughout this experiment.

### 4.6 Version 4 Results — Unroll Last Warp

```
N = 1,048,576  (4.2 MB)
    V4 UnrollWarp    :    0.033 ms  (127.0 GB/s, 39.7% peak)  [OK]
N = 4,194,304  (16.8 MB)
    V4 UnrollWarp    :    0.130 ms  (129.5 GB/s, 40.5% peak)  [OK]
N = 16,777,216  (67.1 MB)
    V4 UnrollWarp    :    0.520 ms  (129.1 GB/s, 40.3% peak)  [OK]
N = 67,108,864  (268.4 MB)
    V4 UnrollWarp    :    1.049 ms  (255.8 GB/s, 79.9% peak)  [OK]
```

**Consolidated bandwidth table (final run, all versions, GB/s):**

| N | V1 | V2 | V3 | V4 | Thrust |
|---|----|----|----|----|--------|
| 1M | 28.4 | 45.8 | 81.9 | 127.0 | 24.9 |
| 4M | 26.2 | 43.0 | 81.6 | 129.5 | 77.5 |
| 16M | 25.6 | 41.8 | 80.4 | 129.1 | 156.0 |
| 64M | 27.8 | 48.3 | 93.1 | **255.8** | 213.8 |

**V4 / V3 incremental speedup:**

| N | V4 / V3 |
|---|---------|
| 1M | 1.55x |
| 4M | 1.59x |
| 16M | 1.61x |
| 64M | **2.75x** |

**The defining feature of V4 — plateau then jump:** V4 sits flat at ~129 GB/s
(40% peak) for N=1M through 16M, then nearly doubles to 256 GB/s (80% peak) at
64M. This plateau-then-jump shape is the most analytically rich result in the
experiment and is explained in full in section 7.6.

At N=64M, V4 reaches **80% of peak bandwidth** and actually **exceeds Thrust**
(256 vs 214 GB/s). State this carefully: V4 is *competitive with Thrust at large
N*, not "faster than Thrust" — Thrust wins at 16M (156 vs 129) and at the small
sizes. Thrust is tuned for robustness across all sizes and data types; V4 is
hand-fit to exactly this problem (float sum, this N, this GPU), so at the largest
size the specialisation edges ahead.

### 4.7 Version 5 Results — Warp Shuffle

Bandwidth (median of 4 confirmed-same-hardware runs; methodology in §4.10):

| N | V4 GB/s | V5 GB/s | V5/V4 |
|---|---|---|---|
| 1M | 129.9 | 135.2 | 1.04x |
| 4M | 138.2 | 149.6 | 1.08x |
| 16M | 129.0 | 142.2 | 1.10x |
| 64M | 220.1 | 253.7 | 1.15x |

Correctness: `[OK]` at every N, every run.

The gain is real (positive in all 4 runs, all N) but small, and
**flat-to-slightly-rising with N** rather than shrinking as N grows — see
§3.6 for why this contradicts the stated hypothesis and §7.7 for the ncu
evidence explaining the shape.

### 4.8 Version 6 Results — Hierarchical Atomic Reduction

Bandwidth (median of 4 confirmed-same-hardware runs):

| N | V5 GB/s | V6 GB/s | V6/V5 |
|---|---|---|---|
| 1M | 135.2 | 139.3 | 1.03x |
| 4M | 149.6 | 146.3 | 0.98x |
| 16M | 142.2 | 143.2 | 1.01x |
| 64M | 253.7 | 254.7 | 1.00x |

Correctness: `[OK]` at every N, every run — the single `atomicAdd` combine
produces results within the same N-scaled tolerance as every other version.

**V6/V5 is statistically flat (0.98-1.03x) — replacing the two-pass
CPU-combine with a single-kernel atomicAdd combine costs nothing
measurable.** See §3.7 and §7.8 for why (block *residency*, not block
*count*, bounds real atomic contention).

**A methodology warning, learned the hard way:** profiling V6 with
`ncu --kernel-name reduce_v6_hierarchical_reduction` on this same binary
produced a program-reported time of **3122 ms** at N=64M — a ~2800x
inflation from the real value. This is not a real regression. `--kernel-name`
re-instruments *every* host-level launch whose name matches — here, all
`WARMUP_RUNS(3) + TIMED_RUNS(10) + 1 correctness call = 14` launches — and
this program's own `GPUTimer` faithfully times that instrumented execution
because it wraps the exact same launch ncu is replaying underneath it. V1-V5
in that same profiled process reported normal times because their kernel
names didn't match the filter and ran unprofiled at native speed. ncu's own
internal `Duration` metric for the profiled kernel read 1.90 ms — consistent
with the real value (~0.93-1.12 ms across runs) once light SOL-section
overhead is accounted for. **Rule going forward: never trust this program's
own printed BenchResult time/bandwidth from a run with ncu attached to a
repeatedly-launched kernel — trust only ncu's own reported Duration and its
ratio/percentage metrics, or use `--launch-skip`/`--launch-count 1` to limit
which single invocation gets the expensive multi-pass treatment.**

### 4.9 Final Consolidated Table — All Versions (Median of 4 Clean Runs)

**Methodology:** 4 independent `./parallel_sum` invocations, same Colab
session, confirmed identical hardware (`Tesla T4, CC 7.5, 40 SMs, 320 GB/s
peak, 8141 GFLOPS peak` printed identically in every run's header). A 5th
run was discarded — it showed V1 at 63.0 GB/s at N=1M (every other run:
24.5-30.5 GB/s) and V4 slower than V3 at N=4M (mechanistically implausible,
since V4 is strictly more optimised than V3) — both signs of an anomalous
run rather than a real measurement. The value shown is the **median** across
the 4 remaining runs at each cell, chosen over the mean for robustness to
the residual single-run flukes documented in §4.10.

Effective bandwidth (GB/s):

| N | CPU | V1 | V2 | V3 | V4 | V5 | V6 | Thrust |
|---|---|---|---|---|---|---|---|---|
| 1M | 1.2 | 27.1 | 43.9 | 84.0 | 129.9 | 135.2 | 139.3 | 22.8 |
| 4M | 1.1 | 29.3 | 47.4 | 89.4 | 138.2 | 149.6 | 146.3 | 62.7 |
| 16M | 1.2 | 25.5 | 41.8 | 80.1 | 129.0 | 142.2 | 143.2 | 147.2 |
| 64M | 1.2 | 26.1 | 76.6 | 146.7 | 220.1 | 253.7 | 254.7 | 207.3 |

% of T4 peak bandwidth (320 GB/s):

| N | V1 | V2 | V3 | V4 | V5 | V6 | Thrust |
|---|---|---|---|---|---|---|---|
| 1M | 8.5% | 13.7% | 26.2% | 40.6% | 42.3% | 43.5% | 7.1% |
| 4M | 9.2% | 14.8% | 27.9% | 43.2% | 46.8% | 45.7% | 19.6% |
| 16M | 8.0% | 13.1% | 25.0% | 40.3% | 44.4% | 44.8% | 46.0% |
| 64M | 8.1% | 23.9% | 45.8% | 68.8% | 79.3% | 79.6% | 64.8% |

**Headline: V5 and V6 both cross into the high-70s% of peak bandwidth at
N=64M — the best of the hand-written kernels — and both beat Thrust
comfortably at every N except 16M**, where Thrust edges ahead by ~3%
(147.2 vs 142-143 GB/s).

**A finding this median data adds that the original single-run notes
missed:** V2 is *not* purely flat across N the way V1 is. V1 stays at
~25-29 GB/s from 1M through 64M — fully sync-bound, immune to scale. V2
rises from ~42-47 GB/s at 1M-16M to 76.6 GB/s at 64M, a ~1.7x jump with no
counterpart in V1. The same cross-block-overlap mechanism documented for V4
in §7.6 (deep enough wave-overlap hides a per-block fixed cost) appears to
partially apply to V2 and V3 as well at 64M — they still carry barriers V4
removed, but enough concurrent blocks are in flight at 131,072-block scale
that *some* of that fixed overhead gets hidden too, just less completely
than for V4/V5/V6. V1 never benefits: its bottleneck (warp divergence in
every barrier step) isn't something cross-block scheduling can hide.

### 4.10 Measurement Reliability on Shared Colab T4

Coefficient of variation (std/mean) across the 4 clean runs:

| Version | CV @ N=16M | CV @ N=64M | Regime at 64M |
|---|---|---|---|
| V1 | 5.3% | 4.2% | fully sync-bound |
| V2 | 5.3% | 17.3% | transitioning |
| V3 | 5.5% | 16.9% | transitioning |
| V4 | 3.9% | 12.9% | **at the plateau→jump threshold** |
| V5 | 3.9% | 6.1% | fully memory-bound |
| V6 | 4.1% | 7.4% | fully memory-bound |
| Thrust | 9.6% | 6.4% | fully memory-bound (library) |

**The pattern is not "large N is noisy" or "memory-bound kernels are
noisy" — it is specifically V2/V3/V4 at N=64M.** At N=16M every version
sits at a uniform, low 4-6% CV. At N=64M, the versions sitting exactly at
the crossover between sync-bound and memory-bound behaviour (V2, V3, and
especially V4 — see §7.6) show 3-4x the variance of versions solidly on
either side of that threshold (V1, always sync-bound; V5/V6/Thrust, solidly
memory-bound at 64M). A kernel balanced right on a regime boundary is
sensitive to small differences in actual achieved cross-block concurrency —
plausibly perturbed run to run on a shared, multi-tenant Colab T4 instance
(co-tenancy, thermal state, dynamic boost clocks) — while a kernel solidly
on one side of that boundary is not.

**Practical consequence:** ratios are far more trustworthy than absolute
GB/s numbers on this hardware. V4/V3 at 64M ranged 1.46x-1.66x across the 4
runs (~7% spread) while V3's own raw GB/s ranged 98.9-160.3 (~48% spread) —
common-mode noise from the shared instance cancels out when comparing two
kernels measured in the *same* run, but not when comparing one kernel's
absolute number *across* runs. Trust the speedup / incremental-speedup
tables more than any single absolute bandwidth number at N≥64M.

---

## 5. PTX Analysis

*Note: PTX section to be filled with actual cuobjdump output when all
versions are complete. The following describes expected structure.*

Both V1 and V2 compile to similar PTX patterns:

**V1 key pattern — predicated add with `@%p` guard:**
```ptx
@%p1 add.f32  %f4, %f3, %f2;    // conditional add — predicated off for idle threads
bar.sync      0;                  // __syncthreads() — appears 8 times
```

**V2 key pattern — same predicated add, different predicate computation:**
```ptx
setp.lt.u32   %p1, %r1, %r2;    // p1 = (tid < s) — contiguous threads active
@%p1 add.f32  %f4, %f3, %f2;    // same predicated add
bar.sync      0;                  // same 8 barriers — unchanged from V1
```

The critical observation: both versions have **identical barrier counts** (8
`bar.sync` instructions). V2 only changes which threads are predicated off,
not how many barriers exist. This explains why sync overhead remains the
dominant bottleneck in both.

---

## 6. ncu Profiling Analysis

### 6.1 V1 vs V2 Side-by-Side Comparison

| Metric | V1 | V2 | Change | Interpretation |
|--------|----|----|--------|---------------|
| Elapsed Cycles | 99,074 | 61,157 | **−38%** | V2 is 38% faster overall |
| Memory Throughput % | 45.06% | 72.89% | +28pp | V2 uses memory subsystem more |
| DRAM Throughput % | 11.77% | 19.10% | +7pp | V2 hits DRAM more often |
| L1/TEX Throughput % | 50.65% | 81.93% | +31pp | V2 uses shared memory more |
| Compute Throughput % | 61.14% | 72.89% | +12pp | V2 computes more efficiently |
| Warp Cycles/Instruction | 11.88 | 20.67 | +74% | V2 slower per instruction |
| Not Predicated Off Threads | 24.46 | 19.91 | **−4.55** | V2 has more predicated-off |
| Executed Instructions | 9,687,040 | 3,264,512 | **−66%** | V2 executes 3× fewer instructions |
| Occupancy | 93.51% | 89.71% | −4pp | slight drop, still excellent |
| ncu Recommendation | Compute > Memory | Compute ≈ Memory | Balanced | V2 more balanced |

### 6.2 The Counterintuitive Predicated-Off Result

V2 has MORE predicated-off threads per warp (19.91 → 12.09 predicated off,
37.8% waste) compared to V1 (24.46 → 7.54 predicated off, 23.6% waste).
This seems to contradict the goal of eliminating divergence.

**The explanation:** ncu averages this metric across ALL instructions in the
kernel. V2 eliminated 6.4 million instructions — specifically the efficient
ones from the first 3 steps where whole warps are active. The remaining
3.3 million instructions are disproportionately from the final 5 steps
(s < 32) where only 1-16 threads per warp are active.

```
V1: 9.7M instructions — mixture of efficient (load phase, first 3 steps)
                         and inefficient (last 5 steps with divergence)
    Average predicated-off: 23.6%

V2: 3.3M instructions — efficient instructions eliminated
                         remaining instructions are the hard ones
    Average predicated-off: 37.8%  ← looks worse because easy work is gone
```

V2 is faster precisely because it eliminated millions of low-efficiency
instructions from V1. The average looks worse because the efficient baseline
instructions have been removed and the hard residual divergence now dominates
the average.

### 6.3 Why Warp Cycles Per Instruction Increased

```
V1: 11.88 cycles per instruction
V2: 20.67 cycles per instruction  ← 74% higher
```

V2 takes nearly twice as many cycles per instruction despite being 38% faster
overall. The reconciliation:

```
V1 total cycle budget: 9.7M × 11.88 = ~115M cycles
V2 total cycle budget: 3.3M × 20.67 =  ~68M cycles
```

V2 does 40% less total cycle-work. Each individual instruction takes longer
in V2 because the sync stall (barrier overhead) is now a larger fraction of
each instruction's execution window — the fast early steps that diluted the
average in V1 are no longer present.

### 6.4 The Shift From Sync-Bound to Balanced

ncu's recommendation changed between V1 and V2:

```
V1: "Compute is more heavily utilised than Memory"
    → sync overhead causing SM to be busy but memory idle

V2: "Compute and Memory are well-balanced"
    → both ~73% utilisation, neither clearly dominant
```

This is genuine progress. V2 has moved from a state where synchronisation
completely dominated to a state where both compute and memory pipelines
are more equally loaded. The remaining bottleneck is now shared between
residual sync overhead (last 5 steps still have barriers) and residual
divergence (last 5 steps have partial warps).

### 6.5 ncu Summary Comparison

| Bottleneck | V1 Severity | V2 Severity | Fix |
|-----------|------------|------------|-----|
| Warp divergence (first 3 steps) | HIGH (23.6% predicated) | ELIMINATED | ✓ Done in V2 |
| Barrier overhead (8 barriers) | HIGH (31.4% stall) | UNCHANGED | → V3, V4 |
| Residual divergence (last 5 steps) | Present | Still present | → V4 |
| Shared memory for last warp | Present | Still present | → V5 |

### 6.6 V3 ncu Analysis — The Memory-Pattern Win

ncu metrics for V3 at N=1M (grid = 2048 blocks, half of V1/V2's 4096):

| Metric | V2 | V3 | Change | Reading |
|--------|----|----|--------|---------|
| Elapsed Cycles | 61,157 | 33,311 | −46% | V3 nearly halves total cycles |
| DRAM Throughput % | 19.10% | 34.06% | +15pp | **the dominant effect** |
| Memory Throughput % | 72.89% | 69.99% | ~same | both balanced |
| Compute Throughput % | 72.89% | 69.99% | ~same | both balanced |
| Memory BW (absolute) | 61 GB/s | 108 GB/s | +77% | wider memory traffic per warp |
| Warp Cycles / Issued Inst | 20.67 | 20.11 | ~same | still stall-heavy |
| Not Predicated Off Threads | 19.91 | 20.81 | +0.9 | divergence ~unchanged |
| Executed Instructions | 3,264,512 | 1,763,328 | −46% | half the blocks → half the work |
| Achieved Occupancy | 89.71% | 89.56% | ~same | warp-slot bound, not resource bound |

**Where the 1.92x actually came from — decomposition:**

The speedup is mostly a **memory-access-pattern win, not a synchronisation win.**
DRAM throughput jumped from 19% (V2) to 34% (V3) — by loading two coalesced
elements per thread, each warp issues wider, denser memory traffic, so the
memory pipeline does more useful work per unit time. Absolute memory bandwidth
rose 61 → 108 GB/s.

The barrier reduction (8 → 7 steps) is real but it is the *minor* contributor:
one fewer barrier out of eight is a ~12% structural change, not a 92% one. The
"first add during load" name undersells what happens — the bigger deal is that
each thread now does twice the memory work before entering the reduction loop.

**What V3 did NOT fix (confirmed by ncu):**

- Warp Cycles Per Issued Instruction stayed at 20.1 (vs V2's 20.67). This high
  value is the proof of continued stalling — on an unstalled kernel it drops to
  single digits. The barriers and residual divergence still dominate.
- Not Predicated Off Threads is 20.81 of 31.82 active — about 11 threads per
  warp still masked off on average. This is the residual divergence in the s<32
  steps, unchanged from V2. V4 targets exactly this.
- The "Compute and Memory well-balanced" at 70% means *neither pipeline is
  saturated* — consistent with a kernel still limited by sync/divergence rather
  than by either compute or memory throughput alone.

### 6.7 How to Read an ncu Report Efficiently (Methodology)

The diagnostic order — triage first, descend only as needed:

```
Step 1 — Speed of Light (always first). Two numbers:
         Memory Throughput % and Compute Throughput %.
           Both high (>80%)     → near a real hardware limit
           One high, one low    → bound by the high one
           Both moderate (~70%) → NEITHER is the limit → descend to Warp State
         V3: both at 70% → bottleneck not visible here → go to Step 2.

Step 2 — Warp State Statistics (when Speed of Light is inconclusive).
         Key metric: Warp Cycles Per Issued Instruction.
           Low (4-8)   → warps issue back-to-back, healthy
           High (15-25)→ warps stall between issues; the stall REASON is
                         the bottleneck (Barrier / Long Scoreboard / Wait)
         V3: 20.1 → heavily stalled → barriers + residual divergence.
         (The per-reason stall breakdown needs --set full or source sampling.)

Step 3 — Predication / Active Threads (for divergence).
         Not Predicated Off Threads Per Warp = 20.8 of 31.8 → ~11 masked.
         That IS the residual divergence. V4 attacks it.

Step 4 — Instruction Statistics (sanity check work done).
         V3: 1.76M instructions vs V2's 3.26M → halved, confirming the
         half-the-blocks structural change took effect.

Step 5 — Occupancy (LAST — only matters if low).
         89.6% → non-issue. Beginners check this first; it should be last,
         since high occupancy is common and rarely the actual problem.
```

Meta-rule: **Speed of Light triages, Warp State diagnoses, the rest confirms.**

### 6.8 V4 ncu Analysis — The Sync→Memory Crossover (profiled at N=64M)

**Methodology note first:** ncu reports "Duration 2.05 ms" for V4 at 64M, but
the cudaEvent timing was 1.05 ms. The 2.05 ms is ncu's *instrumented* time
(13 profiling passes inflate it). Trust cudaEvent for performance; trust ncu's
*ratios and counters* but never its wall-clock duration. Always match the
profiling N to the result being explained — the 64M ncu explains the 64M jump;
a 1M ncu would not.

**Speed of Light (V3 → V4 at 64M):**

| Metric | V3 (1M) | V4 (64M) | Reading |
|--------|---------|----------|---------|
| Memory Throughput % | 70% | 54.7% | both moderate — latency-bound at SM level |
| Compute Throughput % | 70% | 54.7% | balanced with memory |
| DRAM Throughput % | 34% | 50.4% | memory now leads — leaning memory-bound |
| Warp Cycles / Issued Inst | 20.1 | 18.4 | slightly lower stalling |
| Not Predicated Off | 20.8 | 26.9 | more useful work per warp (divergence ↓) |
| Achieved Occupancy | 89.6% | **72.7%** | dropped — the cost of the unroll |

**The headline finding — the stall reason changed category:**

V3's dominant stall was **barrier** (waiting at `__syncthreads()`). V4's ncu now
reports the top stall as:

```
6.1 cycles waiting for a scoreboard dependency on an L1TEX operation
(32.9% of the 18.4 cycles between instruction issues)
```

This is a **memory-load-latency stall**, not a barrier stall. A warp issued a
load through the L1TEX path and its next instruction needs that value, so it
waits. The bottleneck *category* moved from synchronisation to memory latency.

This is the proof — in hardware counters — that removing the 5 barriers worked:
it eliminated the barrier stalls and exposed the *next* layer underneath, which
is memory latency. Peeling one bottleneck reveals the next. That is what
optimisation is.

**Why occupancy dropped (the most instructive counter):**

V4 occupancy fell from 89.6% to 72.7%. Walk through one block's life:

```
Phase 1 (load + loop steps s=128,64):  all 8 warps active → full occupancy
Phase 2 (warpReduce tail, s<32):       if(tid<32) → only warp 0 active
                                        warps 1-7 finished but the BLOCK cannot
                                        release SM resources until ALL its warps
                                        finish → 7 of 8 warp-slots sit idle
                                        during the tail (1/8 = 12.5% local)
```

Averaging Phase 1 (full) with Phase 2 (1/8) lands the kernel at ~73%. **The
unroll-last-warp trick trades barrier stalls for an under-occupied idle-warp
tail.** Net win (V4 is much faster) but the counter shows the price — and this
tail is exactly what drives the bandwidth plateau (section 7.6).

**Instruction count — unrolling removes loop overhead:**

V4 executes fewer instructions per unit work than V3 because the unrolled
`warpReduce` replaced a loop (with its counter increments, comparisons, and
branch instructions) with 6 straight-line adds. Every loop iteration has hidden
non-math instructions; unrolling deletes them. A small but real contributor to
the speedup, easy to miss if only thinking about barriers.

**Cache hit rates ~0% are CORRECT here:** L1 0.69%, L2 1.41%. Reduction touches
each element exactly once — no reuse, nothing to cache. Near-zero hit rate on a
streaming kernel is healthy. (The same metric on matmul would be a disaster,
where reuse is the whole point.) Always interpret a metric against what the
algorithm *should* do.

### 6.9 V6 ncu Analysis — Confirming the Bottleneck Didn't Move

**Methodology warning — read before the numbers:** this program's own
printed time for V6 in the profiling run was 3122 ms, a ~2800x inflation
from ncu re-instrumenting all 14 host-level launches of the matched kernel
name (3 warmup + 10 timed + 1 correctness call). That number is discarded
entirely — full explanation in §4.8. Everything below uses ncu's own
internal per-pass metrics, which are normalised/relative and not subject to
that inflation.

**Speed of Light (profiled at N=64M):**

| Metric | Value | Reading |
|--------|-------|---------|
| Memory Throughput | 53.6% | moderate |
| DRAM Throughput | 53.6% | moderate |
| Compute (SM) Throughput | 51.3% | moderate, balanced with memory |
| ncu's own note | — | "below 60% typically indicates latency issues" |

Both pipelines sit around 51-54% — neither saturated. ncu's own diagnostic
points at latency, not a raw bandwidth or compute ceiling.

**Warp State Statistics:**

| Metric | V3 (old) | V4 (old) | V6 | Reading |
|--------|----|----|----|---------|
| Warp Cycles Per Issued Instruction | 20.1 | 18.4 | **19.88** | unchanged within noise across V3→V4→V6 |
| Avg. Active Threads Per Warp | — | — | 31.77 | near-full warps |
| Avg. Not Predicated Off Threads | 20.8 | 26.9 | **26.41** | matches V4, not further improved |

**This is the key finding: V6 (warp shuffle + atomic combine) did not move
the needle on warp-level stalling relative to V4.** If shuffle and the
atomic combine were fixing the dominant bottleneck, this number should have
dropped meaningfully. It didn't.

**Stall reason — still long-scoreboard, still ~35%:**

> "On average, each warp of this workload spends 6.9 cycles being stalled
> waiting for a scoreboard dependency on a L1TEX (local, global, surface,
> texture) operation... about 34.7% of the total average of 19.9 cycles."

This is specifically a **long-scoreboard** stall — waiting on the L1TEX path
(global/local/texture/surface memory round trips) — not a **short-scoreboard**
stall (which would mean waiting on shared-memory/MIO-path operations). V4's
old analysis (§6.8) found the same stall category at nearly the same
magnitude (~33%). Neither warp shuffle (touches only shared-memory traffic
in the last warp) nor the atomic combine (one instruction at the very end
of the kernel) had reason to move a stall that originates in the
global-load / early tree-reduction phase — and the counters confirm they
didn't. See §7.7 for what this means for the V5 hypothesis.

**Occupancy:**

| Metric | V4 (old) | V6 |
|--------|----------|-----|
| Theoretical | 100% | 100% |
| Achieved | 72.7% | **75.0%** |

A small improvement over V4 — plausibly the unrolled shuffle sequence having
fewer instructions in the tail than the volatile-shared-memory version,
slightly shortening the low-occupancy tail window. Not the dominant effect.

**One ncu suggestion to ignore:** ncu's OPT note flags "4,849,664 non-fused
FP32 instructions... converting pairs to FMA could increase performance up
to 50%." **Not applicable here** — this is a pure-sum reduction with no
multiply operand anywhere in the kernel, so there is nothing to fuse into an
FMA. This is a generic advisory ncu attaches to any kernel with many plain
FADD instructions; it isn't a real opportunity for this algorithm.

**Cache hit rates (L1 0%, L2 1.6%) are correct, not a problem** — same
reasoning as V4's analysis (§6.8): a streaming reduction touches each
element exactly once, so near-zero reuse is the expected, healthy signature.

---

## 7. Interpretation

### 7.1 What V2 Fixed — And What It Did Not

**Fixed:** Warp divergence for the first 3 reduction steps (s=128, 64, 32).
At these steps, active threads are contiguous (0 to s-1), whole warps are
either fully active or fully idle, and no warp has a mixed split. The hardware
executes full-warp instructions with no predication waste. This is where the
1.63x speedup comes from.

**Not fixed:** Eight `__syncthreads()` barriers per block — identical to V1.
The sync stall that consumed 31.4% of V1's cycles is still present in V2.
Reducing the barrier count (V3, V4) and eventually eliminating barriers for
the last warp entirely (V5) are the required next steps.

**Not fixed:** Residual divergence in the last 5 steps (s=16, 8, 4, 2, 1).
When s < 32, only s threads are active within the final 32-thread warp. Thread
0 through s-1 are active; threads s through 31 are idle. This intra-warp
divergence is structurally the same as V1's divergence — just limited to the
final warp. V4 (unroll last warp) eliminates this by removing the conditional
for the last 5 steps entirely.

### 7.2 Why Flat Bandwidth — The Sync-Bound Signature

Both V1 (~27 GB/s) and V2 (~42 GB/s) show flat bandwidth across all N.
The explanation:

Each block always executes exactly 8 `__syncthreads()` barrier steps
regardless of N. As N grows, more blocks run in parallel but each block
takes the same time. The per-block sync overhead is a fixed cost that does
not decrease with N, so total throughput does not increase with N.

Thrust scales from 24 GB/s to 210 GB/s because it uses warp shuffle
throughout — zero `__syncthreads()` calls. As N grows, Thrust has more
parallel work with no synchronisation bottleneck, and the memory controller
stays saturated. The bandwidth scales because nothing stalls the pipeline.

The transition from sync-bound to memory-bound is precisely what V3 through
V5 achieve — each version reduces barrier count, and the bandwidth curve
begins to slope upward with N rather than staying flat.

### 7.3 Why V2 Bandwidth Is Still Far From Peak

V2 achieves ~42 GB/s = 13% of 320 GB/s peak. The theoretical minimum time
at N=16M is 0.2 ms (320 GB/s). V2 takes 1.6 ms — 8× slower than the peak.

The gap is entirely barrier overhead. Each of the 8 barriers stalls all
32 warps in a block until the slowest warp arrives. With 65,536 blocks
across 40 SMs, the SM is never idle — but within each block, 31.4% of
cycles are spent waiting at barriers rather than doing useful work.

If barriers were free, V2 would run at ~42 × (1 / (1 - 0.314)) = ~61 GB/s.
If barriers AND residual divergence were eliminated, the kernel would approach
the memory bandwidth ceiling. This is what the V3-V5 progression achieves.

### 7.4 Connection to CPU Lab

The progression from V1 to V2 mirrors exp2 of the CPU lab (branch prediction):

```
CPU exp2:  sorted array eliminated branch mispredictions → 8.4× speedup
           but the branch itself still existed (just predicted correctly)

GPU V2:    sequential addressing eliminated warp divergence → 1.63× speedup
           but the __syncthreads() barrier still exists (just cleaner)
```

In both cases, fixing the branch/divergence pattern improved performance
significantly but the next bottleneck (pipeline stalls / sync barriers)
was immediately exposed as the next target.

### 7.5 The V3 → V5 Path

```
V2 remaining problems:
    8 __syncthreads() per block     → 31.4% stall overhead
    Last 5 steps divergent          → residual predicated-off waste

V3 (first add during load):
    Thread loads 2 elements, adds before entering loop
    Halves the problem per block → 7 steps, 7 barriers
    Also doubles useful work per global memory load

V4 (unroll last warp):
    s < 32 → only 1 warp active → __syncthreads() unnecessary
    Removes 5 barriers: 8 → 3 remaining
    Unrolled code has no predicate → residual divergence gone

V5 (warp shuffle):
    Last warp uses __shfl_down_sync instead of shared memory
    Zero barriers for last 5 steps
    Register-to-register communication, no bank conflicts
    All threads in the warp participate → no predication

After V5: only 3 barriers remain (first 3 steps)
Expected bandwidth: approaching Thrust territory
Remaining gap vs Thrust: vectorised loads (float4) not yet implemented
```

### 7.6 The Bandwidth Plateau-Then-Jump — V4's Defining Behaviour

This is the richest single result in the experiment. V4's bandwidth:

```
N=1M    127 GB/s
N=4M    129 GB/s   ← plateau
N=16M   129 GB/s   ← plateau
N=64M   256 GB/s   ← jump to 80% peak
```

**The model: total_time = streaming_time + structural_overhead.**

```
T_stream  = (N × 4) / achievable_DRAM_BW   → grows LINEARLY with N
T_tail    = under-occupied warpReduce tail + launch/drain overhead
          → roughly FIXED fraction of each block's life
```

**Why the plateau (1M–16M):** To hit peak DRAM bandwidth the GPU needs many
concurrent memory requests in flight to hide the ~400-600 cycle DRAM latency
(Little's Law: bytes-in-flight = bandwidth × latency). During V4's warpReduce
tail, only 1 of 8 warps per block is active, and it issues *shared*-memory ops,
not *global* loads. So during the tail, global-memory requests in flight drop,
the DRAM bus goes partially idle, and average bandwidth falls below peak. The
tail is a *fixed fraction* of each block's life, so this caps bandwidth at
~129 GB/s regardless of N across 1M–16M. The plateau is a **structural ceiling
set by the occupancy tail, not by N.**

**Why the jump at 64M:** The tail fraction per block is unchanged — so why does
64M escape the ceiling? Two effects combine:

1. **Cross-block overlap.** At 64M, V4 launches 131,072 blocks for 160 resident
   slots (40 SMs × 4) — roughly 800 waves of blocks cycling through. With that
   depth, while block X is in its under-occupied tail on an SM, block Y (just
   arrived) is in its full-occupancy load phase on the same SM, issuing global
   loads. The tails of some blocks overlap the load-phases of others, keeping
   aggregate global-memory requests high. At small N there aren't enough waves
   for this overlap to develop — the machine visibly drains and refills.

2. **Fixed overhead becomes negligible.** Launch latency and final-wave drain
   are a meaningful fraction of a 0.5 ms run but vanish in a longer one. At 64M
   the streaming time dominates so completely these constants disappear.

So the jump is **the point where cross-block overlap becomes deep enough to hide
the per-block occupancy tail, and fixed overheads become negligible.**

**The elegant confirmation — effective BW (80%) exceeds ncu's per-SM DRAM
throughput (50%).** How can system effective bandwidth outrun the per-SM
measurement? Because the DRAM bus is a *shared global resource*. ncu's 50%
DRAM throughput is the per-SM view; 40 SMs each at 50% local memory throughput
collectively saturate one memory controller. The system-level effective
bandwidth (total bytes / wall-clock = 80% peak) therefore exceeds any single
SM's local reading. This resolves the apparent contradiction between the 80%
effective number and the 50% ncu number — they measure different scopes.

**Why this matters for V5:** At 64M, V4 is already at 80% of the memory ceiling.
The roofline model says: once you are at the memory wall, further *compute*
optimisation is wasted. V5 (warp shuffle) is a compute/latency win — it removes
the volatile shared-memory round-trips in the last warp. So the explicit
hypothesis for V5:

```
V5 helps at SMALL N (latency-bound — fewer dependent shared-mem ops in the tail)
V5 barely moves LARGE N (already memory-bound at 80% peak — no headroom)
```

If confirmed, this is itself a valuable finding: the roofline model predicting
that compute optimisation stops paying off once the memory wall is reached.

### 7.7 V5's Hypothesis, Falsified — And What Actually Explains It

The hypothesis above predicted V5 would help most at small N and barely move
large N, reasoning that V4 is already memory-bound (80% of peak) at 64M, so
compute/latency wins should have no headroom left there.

**Measured (median of 4 runs): V5/V4 = 1.04x (1M), 1.08x (4M), 1.10x (16M),
1.15x (64M) — flat-to-rising, the opposite of the predicted shape.**

The hypothesis's reasoning was sound but aimed at the wrong bottleneck. It
assumed the memory wall observed in *aggregate* bandwidth (§7.6) was the
binding constraint everywhere in the kernel, so removing a small last-warp
cost would have nothing left to win once that wall is hit. But ncu on V6
(§6.9, sharing V5's shuffle tail) shows the actual dominant stall — 34.7%,
long-scoreboard, waiting on L1TEX/global-memory round trips — sits in the
*load / early tree-reduction* phase, not the last warp. Being "memory-bound
in aggregate" doesn't mean every code region is equally memory-bound; the
last-warp shared-memory traffic shuffle replaced was never the dominant
cost at any N, so removing it delivers a similarly small, roughly
N-independent win everywhere — exactly the flat-to-slightly-rising shape
measured. The slight rise with N is more likely a secondary effect of the
same cross-block-overlap mechanism from §7.6 (more blocks in flight at
large N means the last-warp savings are also better hidden/amortised) than
the mechanism the original hypothesis proposed.

**The general lesson:** a roofline-level "memory-bound" verdict describes
the kernel in aggregate; it does not say where inside the kernel the
remaining stall time actually lives. That requires the warp-state/stall-
reason breakdown, not the Speed-of-Light section alone — precisely the
diagnostic order §6.7 already lays out.

### 7.8 Why the Hierarchical Atomic Combine Is Free

V6 replaces V5's per-block `partial_sums[blockIdx.x] = val` (one plain
write) with `atomicAdd(output, val)` (one atomic read-modify-write to a
single, block-shared address). Naively, funnelling 131,072 blocks'-worth of
atomics (at N=64M) into one 4-byte location sounds like a serialisation
bottleneck. Measured, it costs 0.98-1.03x — indistinguishable from noise.

The reason is occupancy math, the same kind used throughout §7.6: only
`40 SMs × 4 resident blocks/SM = 160` blocks are actually executing at any
given instant, not all 131,072 simultaneously. Contention on the output
address is bounded by however many of those ~160 blocks happen to finish
their local reduction and reach the atomicAdd at the same moment — a much
smaller, naturally staggered number, not the full block count. Turing's L2
cache also accelerates atomics with a hardware combining path for exactly
this pattern (many RMWs to one resident cache line). The single atomic
instruction per block is a vanishingly small fraction of each block's own
work (a 512-element local reduction), so even a nonzero atomic cost would
be hard to see above run-to-run noise (§4.10).

**Practical implication:** for reductions where the combine step would
otherwise require a second kernel launch, a D2H copy, or host-side code,
the atomic-combine pattern is close to free on modern hardware as long as
block *residency* (not total block *count*) stays reasonable. The classic
worry about "atomic contention" applies far more to designs where many
*threads within a single wave* hit the same address (e.g. one atomicAdd
per thread instead of per block), not to this one-atomic-per-block pattern.

---

## 8. Key Takeaways (V1 through V6)

**V1 is sync-bound, not memory-bound.** Despite the roofline model predicting
memory-bound behaviour, V1 achieves only 8% of peak bandwidth because 31.4%
of its cycles are spent waiting at `__syncthreads()` barriers. Data is loaded
from DRAM once into shared memory — all subsequent work happens on-chip. The
roofline model predicts the ceiling but not the intermediate bottleneck.

**V2 eliminates divergence for the first 3 steps — 1.63x consistent speedup.**
Sequential addressing ensures whole warps are either fully active or fully idle.
No warp has a mixed active/idle split for steps where s ≥ 32. The 1.63x
improvement is consistent across all N — confirming it is a fixed per-block
gain, not a scaling improvement.

**The counterintuitive ncu finding: V2 shows more predicated-off waste than V1.**
This is because V2 eliminated 6.4 million efficient instructions (from the first
3 divergence-free steps), leaving the trace dominated by the remaining hard
instructions (last 5 steps with residual divergence). The average looks worse
because the easy work is gone. The kernel is faster precisely because of this
elimination.

**Both V1 and V2 have flat bandwidth — they are sync-bound.**
Flat bandwidth across all N is the diagnostic signature of a sync-bound kernel.
Memory-bound kernels scale with N (as Thrust demonstrates). The transition to
scaling behaviour requires eliminating barriers (V3, V4, V5).

**The shift from sync-dominant to balanced.** ncu's classification changed from
"Compute > Memory" (V1) to "Compute ≈ Memory" (V2). Both pipelines are now at
~73% utilisation. This is genuine progress — but both are still below peak
because the remaining sync and divergence bottlenecks prevent either pipeline
from reaching full utilisation simultaneously.

**Barrier count is unchanged between V1 and V2 — 8 barriers per block.**
V2's improvement came entirely from divergence elimination, not from reducing
synchronisation. The next three versions (V3, V4, V5) systematically reduce
barriers from 8 to 3 and eventually eliminate them for the final warp entirely.

**V3 — the win was memory coalescing, not the barrier saving.** Loading two
coalesced elements per thread raised DRAM throughput 19→34%; the single-barrier
reduction was the minor contributor. V3 bandwidth first slopes upward with N
(52→158 GB/s), the first break from the flat sync-bound signature.

**V4 — the sync→memory crossover, confirmed in counters.** Unrolling the last
warp flipped the dominant stall reason from barrier (V3) to L1TEX memory latency
(V4). The cost was an idle-warp tail dropping occupancy 89→73%. The result is
the plateau-then-jump bandwidth curve: flat at 129 GB/s through 16M, then 256
GB/s (80% peak) at 64M once cross-block overlap hides the tail (section 7.6).

**The general lesson across V1→V4: optimisation peels bottlenecks in layers.**
Each version removed the current binding constraint and exposed the next:
divergence (V1→V2) → memory access pattern (V2→V3) → barriers (V3→V4) → memory
latency (the current V4 ceiling). The ncu stall-reason metric is what makes each
layer visible.

**V5 — a real but small win, and a falsified hypothesis.** Warp shuffle
replaced V4's volatile-shared-memory last-warp reduction, gaining a
consistent but modest 1.04-1.15x across N — flat-to-rising, not the
small-N-only pattern predicted in §7.6. ncu confirms why: the dominant
stall (long-scoreboard, ~35%, unchanged from V4) lives in the load/early
tree-reduction phase, a region shuffle never touches. Being "memory-bound
in aggregate" does not mean every part of the kernel is memory-bound — the
layers-peel-one-at-a-time lesson from V1→V4 has a corollary: not every
remaining layer is reachable by every optimisation.

**V6 — the hierarchical atomic combine is architecturally superior at zero
cost.** Replacing the two-pass CPU combine (D2H copy + host loop) with a
single `atomicAdd` into one global scalar measured at 0.98-1.03x of V5 —
statistically free. Block *residency* (~160 concurrent blocks on this GPU),
not total block *count* (131,072 at 64M), bounds real atomic contention,
which is why this pattern scales without a measurable tax. V5 and V6 both
now sit at ~79-80% of T4 peak bandwidth and beat Thrust at every N except
16M.

**A new methodological lesson from repeated runs: variance is a threshold
phenomenon, not a large-N phenomenon.** Re-running the full sweep multiple
times on the same (confirmed identical) Colab T4 showed uniform 4-6% CV
across all versions at N=16M, but 13-17% CV specifically for V2, V3, and V4
at N=64M — the versions sitting at or near the plateau→jump crossover
(§7.6) — while V1 (always sync-bound) and V5/V6/Thrust (solidly
memory-bound at 64M) stayed at 4-7% CV even at 64M. A kernel balanced on a
regime boundary is fragile to small run-to-run differences in achieved
concurrency; one solidly on either side is not. Ratios between kernels
measured in the same run are far more trustworthy than any single kernel's
absolute bandwidth compared across runs (§4.10).

---

## 9. Versions Implemented — Status

| Version | Change | Target Metric | Result |
|---------|--------|--------------|--------------|
| V3 | First add during load | DRAM throughput 19→34%, bandwidth↑ | ✓ DONE — 1.92x over V2 |
| V4 | Unroll last warp | Stall flips barrier→memory; occupancy 89→73% | ✓ DONE — 1.44-1.66x over V3 at 64M (median of 4 runs) |
| V5 | Warp shuffle for last warp | Removes volatile shared-mem round-trips | ✓ DONE — 1.04-1.15x over V4, flat-to-rising with N (hypothesis falsified — §7.7) |
| V6 | Hierarchical atomic reduction | Single-kernel combine, no host pass | ✓ DONE — 0.98-1.03x of V5, statistically free (§7.8) |
| Thrust comparison | Library baseline | Full pipeline | V5/V6 beat Thrust at every N except 16M |

At N=64M (median of 4 runs), V5/V6 reach **~79-80% of peak bandwidth**,
exceeding Thrust (254 vs 207 GB/s). Both hypotheses formed after V4 (§7.6's
V5 prediction, and the atomic-contention prediction in §3.7) were
directionally wrong but usefully so — see §7.7 and §7.8 for what the ncu
counters revealed instead.

**Not implemented — deliberately out of scope for now:**
- General thread coarsening (`COARSE_FACTOR` swept as a parameter, rather
  than V3/V4/V5/V6's fixed factor of 2). This is PMPP's own generalisation
  of what V3 already does in miniature; worth a dedicated V7 if revisited.
- Vectorised loads (`float4`) to cut load-instruction count — flagged as
  remaining headroom back in the V4 write-up, still open.
- Bank-conflict measurement via ncu's dedicated shared-memory counters —
  the bank-conflict analysis in problem.md §4 remains conceptual only,
  never measured directly.