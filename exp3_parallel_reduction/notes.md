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

### 3.5 Version 4 — Unroll Last Warp

**Predicted speedup over V3: 1.1-1.3x.**
Removes 5 `__syncthreads()` calls. Constant improvement regardless of N.

### 3.6 Version 5 — Warp Shuffle

**Predicted speedup over V4: 1.2-1.5x.**
No shared memory for last warp, no barriers for final 5 steps.

### 3.7 Thrust Reference

**Predicted: V5 reaches 50-70% of Thrust bandwidth.**

### 3.8 Predicted Summary Table

| Version | Key Mechanism | Predicted BW | % of Peak |
|---------|-------------|-------------|-----------|
| CPU | Serial accumulator | ~50 GB/s | N/A |
| V1 | Warp divergence | 50-100 GB/s | 15-30% |
| V2 | No divergence (first 3 steps) | 100-150 GB/s | 30-47% |
| V3 | Half sync barriers | 130-180 GB/s | 40-56% |
| V4 | No barriers last warp | 150-200 GB/s | 47-62% |
| V5 | Warp shuffle | 180-220 GB/s | 56-69% |
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

---

## 8. Key Takeaways (V1 and V2)

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

---

## 9. Versions Still To Be Implemented

| Version | Change | Target Metric | Expected Gain |
|---------|--------|--------------|--------------|
| V3 | First add during load | Barrier count 8→7, bandwidth↑ | 1.3-1.7× over V2 |
| V4 | Unroll last warp | Barrier count 7→3, divergence→0 | 1.1-1.3× over V3 |
| V5 | Warp shuffle for last warp | Zero barriers last 5 steps | 1.2-1.5× over V4 |
| Thrust comparison | Library baseline | Full pipeline | V5 → 50-70% of Thrust |