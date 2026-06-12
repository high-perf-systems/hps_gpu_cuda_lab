// ============================================================
// Experiment 3: Parallel Reduction
// hps_gpu_cuda_lab — github.com/high-perf-systems
//
// Versions implemented in this file:
//   1. CPU baseline   (sequential sum — serial dependency chain)
//   2. GPU V1         (interleaved addressing — warp divergence present)
//   3. GPU V2         (sequential addressing — divergence eliminated)
//   4. thrust::reduce (library reference baseline)
//
// Versions to be added:
//   5. GPU V3         (first add during load — halve sync barriers)
//   6. GPU V4         (unroll last warp — remove 5 barriers)
//   7. GPU V5         (warp shuffle — no shared memory for last warp)
//
// Build:
//   nvcc -O2 -o parallel_reduction parallel_reduction.cu -lm
//
// Run all sizes:
//   ./parallel_reduction
//
// Run single size:
//   ./parallel_reduction 16777216
//
// Profile timeline (works on T4):
//   nvprof ./parallel_reduction 16777216
//
// Profile hardware metrics (ncu required on CC 7.5+):
//   ncu --section SpeedOfLight \
//       --section WarpStateStats \
//       --section InstructionStats \
//       --section MemoryWorkloadAnalysis \
//       --section Occupancy \
//       --kernel-name reduce_v2_sequential \
//       ./parallel_reduction 16777216 2>&1 | head -200
//
// Key ncu metrics:
//   Avg. Not Predicated Off Threads Per Warp  (divergence waste)
//   smsp__warp_issue_stalled_barrier_*        (sync stall %)
//   DRAM Throughput %                         (memory bound check)
//   Compute Throughput %                      (compute bound check)
//   Achieved Occupancy %                      (warp slot utilisation)
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

// ============================================================
// ERROR CHECKING
// ============================================================
#define CUDA_CHECK(call)                                        \
do {                                                            \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d -- %s\n",         \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while (0)

// ============================================================
// CONFIG
// ============================================================
#define BLOCK_SIZE     256      // threads per block — all versions
#define WARMUP_RUNS      3      // discarded before timing
#define TIMED_RUNS      10      // averaged for reported time

#define T4_PEAK_BW_GBS   320.0f   // T4 peak DRAM bandwidth (GB/s)
#define T4_PEAK_GFLOPS  8141.0f   // T4 peak FP32 compute (GFLOPS)

// ============================================================
// RESULT STRUCT
// ============================================================
typedef struct {
    int   N;
    float time_ms;        // average kernel time (warmup discarded)
    float bandwidth_GBs;  // N * sizeof(float) / time
    float pct_peak_bw;    // bandwidth as % of T4 peak (320 GB/s)
} BenchResult;

// ============================================================
// GPU TIMER — cudaEvent based, same as matmul.cu
// ============================================================
struct GPUTimer {
    cudaEvent_t start, stop;
    GPUTimer()  {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GPUTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    void Start() { CUDA_CHECK(cudaEventRecord(start, 0)); }
    void Stop()  {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
    }
    float ElapsedMs() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ============================================================
// CPU TIMER — monotonic wall clock, same as matmul.cu
// ============================================================
static inline double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================
// BANDWIDTH HELPER
// Reduction reads N floats from DRAM, writes 1 scalar back.
// The write is negligible, so effective bandwidth = N * 4 / time.
// ============================================================
static inline float compute_bw(int N, float time_ms) {
    double bytes   = (double)N * sizeof(float);
    double seconds = time_ms / 1000.0;
    return (float)(bytes / seconds / 1e9);
}

// ============================================================
// CORRECTNESS CHECK
// Different summation orders produce different floating-point
// rounding. Tolerance is scaled by N (error accumulates with N).
// ============================================================
bool verify(float cpu_result, float gpu_result, int N) {
    float eps  = (float)N * 1e-5f;
    float diff = fabsf(cpu_result - gpu_result);
    float mag  = fabsf(cpu_result) + 1e-6f;
    if (diff / mag > eps) {
        printf("  MISMATCH: cpu=%.6f  gpu=%.6f  rel_err=%.2e  tol=%.2e\n",
               cpu_result, gpu_result, diff / mag, eps);
        return false;
    }
    return true;
}

// ============================================================
// TWO-PASS HOST REDUCTION
// Each kernel produces one partial sum per block in d_partial.
// This function copies them to CPU and sums them.
// For BLOCK_SIZE=256 and N=16M: 65,536 partial sums.
// CPU reduction of 65,536 floats takes ~0.02ms — negligible.
// ============================================================
float reduce_partial_sums_on_cpu(float* d_partial, int grid_size) {
    float* h_partial = new float[grid_size];
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial,
                          grid_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < grid_size; i++) sum += h_partial[i];
    delete[] h_partial;
    return sum;
}

// ============================================================
// CPU BASELINE
// Single-threaded sequential sum. IEEE 754 non-associativity
// prevents the compiler from reordering additions without
// -ffast-math, so this is a strict serial dependency chain:
// each add must wait for the previous result.
// ============================================================
float reduce_cpu(const float* A, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        sum += A[i];
    return sum;
}

// ============================================================
// GPU V1 — INTERLEAVED ADDRESSING
//
// The stride s starts at 1 and doubles each step.
// Active threads have indices that are multiples of 2*s:
//   s=1:   threads 0,2,4,6...    active — stride 2
//   s=2:   threads 0,4,8,12...   active — stride 4
//   s=128: thread 0 only active
//
// Problem: within each warp of 32 threads, some are active and
// some are idle — a DIVERGENT warp. The hardware must execute
// both the active path (with idle threads masked) and the idle
// path (with active threads masked). At s=1, only 16 of 32
// threads per warp do useful work = 50% efficiency.
// Divergence compounds: s=2 → 25%, s=4 → 12.5%...
//
// Average active thread fraction across 8 steps:
//   (128+64+32+16+8+4+2+1) / (256×8) = 255/2048 = 12.5%
//   → 87.5% of instruction slots wasted on idle threads
// ============================================================
__global__ void reduce_v1_interleaved(
    const float* __restrict__ A,
    float*       __restrict__ partial_sums,
    int N)
{
    extern __shared__ float sdata[];

    int tid        = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Load one element per thread — out-of-bounds threads load 0
    sdata[tid] = (global_idx < N) ? A[global_idx] : 0.0f;
    __syncthreads();   // all threads must finish loading before any reads

    // Tree reduction — interleaved addressing (causes warp divergence)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {       // only strided threads active
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();   // barrier: all threads finish step before next
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

// ============================================================
// GPU V2 — SEQUENTIAL ADDRESSING
//
// The stride s starts at blockDim.x/2 and halves each step.
// Active threads are always 0..s-1 — the FIRST s threads.
//
// Key difference from V1: active threads are CONTIGUOUS.
// Each warp is either FULLY ACTIVE (warp index < s/32) or
// FULLY IDLE (warp index >= s/32). No warp ever has a mixed
// active/idle split — divergence is eliminated for the first
// log2(blockDim.x) - log2(32) = 8 - 5 = 3 steps.
//
// When s < 32 (the last 5 steps), the remaining threads fit
// in a single warp. That warp has mixed active/idle slots again
// — divergence reappears for the final 5 steps. This residual
// divergence is what V4 (unroll last warp) eliminates.
//
// V2 improvement over V1:
//   - First 3 steps: full warps active, zero divergence
//   - Steps 4-8: one warp, some divergence (same as V1 final steps)
//   - Net effect: divergence predicated-off drops from 23.6% to ~5-8%
// ============================================================
__global__ void reduce_v2_sequential(
    const float* __restrict__ A,
    float*       __restrict__ partial_sums,
    int N)
{
    extern __shared__ float sdata[];

    int tid        = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Load one element per thread — identical to V1
    sdata[tid] = (global_idx < N) ? A[global_idx] : 0.0f;
    __syncthreads();

    // Tree reduction — sequential addressing (no divergence for first steps)
    // s=128: threads 0-127 active, threads 128-255 idle → 4 full warps active
    // s=64:  threads 0-63  active, rest idle             → 2 full warps active
    // s=32:  threads 0-31  active, rest idle             → 1 full warp active
    // s=16:  threads 0-15  active, rest idle             → half warp (diverges)
    // s=8:   threads 0-7   active ...
    // s=4, s=2, s=1: progressively more idle in final warp
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {                  // first s threads active — contiguous
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

// ============================================================
// BENCHMARK: CPU
// ============================================================
BenchResult benchmark_cpu(const float* A, int N) {
    for (int r = 0; r < WARMUP_RUNS; r++)
        (void)reduce_cpu(A, N);

    double total = 0.0;
    float  result = 0.0f;
    for (int r = 0; r < TIMED_RUNS; r++) {
        double t0 = now_ms();
        result = reduce_cpu(A, N);
        total += now_ms() - t0;
    }
    (void)result;

    float avg_ms = (float)(total / TIMED_RUNS);
    BenchResult br;
    br.N             = N;
    br.time_ms       = avg_ms;
    br.bandwidth_GBs = compute_bw(N, avg_ms);
    br.pct_peak_bw   = 0.0f;
    return br;
}

// ============================================================
// BENCHMARK: GPU KERNEL — generic helper
// Takes a kernel function pointer so V1 and V2 share the same
// timing infrastructure. The kernel signature is fixed:
//   (const float* A, float* partial, int N)
// with shared memory size = BLOCK_SIZE * sizeof(float).
// ============================================================
typedef void (*ReduceKernel)(const float*, float*, int);

BenchResult benchmark_gpu(
    const float*  d_A,
    int           N,
    ReduceKernel  kernel)
{
    int   grid_size  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int   smem_bytes = BLOCK_SIZE * sizeof(float);
    float result     = 0.0f;

    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

    GPUTimer timer;

    // Warmup — ensures GPU is at steady-state frequency
    for (int r = 0; r < WARMUP_RUNS; r++) {
        kernel<<<grid_size, BLOCK_SIZE, smem_bytes>>>(d_A, d_partial, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        result = reduce_partial_sums_on_cpu(d_partial, grid_size);
    }

    // Timed runs — cudaEvent timing excludes host-side partial sum reduction
    float total_ms = 0.0f;
    for (int r = 0; r < TIMED_RUNS; r++) {
        timer.Start();
        kernel<<<grid_size, BLOCK_SIZE, smem_bytes>>>(d_A, d_partial, N);
        timer.Stop();
        total_ms += timer.ElapsedMs();
        if (r == TIMED_RUNS - 1)
            result = reduce_partial_sums_on_cpu(d_partial, grid_size);
    }
    (void)result;

    CUDA_CHECK(cudaFree(d_partial));

    float avg_ms = total_ms / TIMED_RUNS;
    BenchResult br;
    br.N             = N;
    br.time_ms       = avg_ms;
    br.bandwidth_GBs = compute_bw(N, avg_ms);
    br.pct_peak_bw   = br.bandwidth_GBs / T4_PEAK_BW_GBS * 100.0f;
    return br;
}

// ============================================================
// BENCHMARK: THRUST REFERENCE
// ============================================================
BenchResult benchmark_thrust(const float* d_A, int N) {
    thrust::device_ptr<const float> ptr(d_A);
    GPUTimer timer;
    float result = 0.0f;

    for (int r = 0; r < WARMUP_RUNS; r++)
        result = thrust::reduce(ptr, ptr + N, 0.0f, thrust::plus<float>());

    float total_ms = 0.0f;
    for (int r = 0; r < TIMED_RUNS; r++) {
        timer.Start();
        result = thrust::reduce(ptr, ptr + N, 0.0f, thrust::plus<float>());
        timer.Stop();
        total_ms += timer.ElapsedMs();
    }
    (void)result;

    float avg_ms = total_ms / TIMED_RUNS;
    BenchResult br;
    br.N             = N;
    br.time_ms       = avg_ms;
    br.bandwidth_GBs = compute_bw(N, avg_ms);
    br.pct_peak_bw   = br.bandwidth_GBs / T4_PEAK_BW_GBS * 100.0f;
    return br;
}

// ============================================================
// PRINT SUMMARY TABLES
// One column per version. Expanding: add column header + data
// for each new version as it is implemented.
// ============================================================
void print_summary(
    const BenchResult* cpu,
    const BenchResult* v1,
    const BenchResult* v2,
    const BenchResult* thr,
    const int*         Ns,
    int                num_sizes)
{
    printf("\n");
    printf("================================================================\n");
    printf("SUMMARY — %d timed runs (after %d warmup)\n",
           TIMED_RUNS, WARMUP_RUNS);
    printf("T4 Peak BW: %.0f GB/s\n", T4_PEAK_BW_GBS);
    printf("================================================================\n");

    // ---- Table 1: Time (ms) ----
    printf("\n[1] KERNEL TIME (ms)\n");
    printf("%-12s %10s %12s %12s %12s\n",
           "N", "CPU", "V1 Interl.", "V2 Sequen.", "Thrust");
    printf("%-12s %10s %12s %12s %12s\n",
           "----------", "--------", "----------", "----------", "------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-12d %10.3f %12.4f %12.4f %12.4f\n",
               Ns[i],
               cpu[i].time_ms,
               v1[i].time_ms,
               v2[i].time_ms,
               thr[i].time_ms);
    }

    // ---- Table 2: Bandwidth (GB/s) ----
    printf("\n[2] EFFECTIVE BANDWIDTH (GB/s)  —  T4 Peak: %.0f GB/s\n",
           T4_PEAK_BW_GBS);
    printf("%-12s %10s %12s %12s %12s\n",
           "N", "CPU", "V1 Interl.", "V2 Sequen.", "Thrust");
    printf("%-12s %10s %12s %12s %12s\n",
           "----------", "--------", "----------", "----------", "------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-12d %10.1f %12.1f %12.1f %12.1f\n",
               Ns[i],
               cpu[i].bandwidth_GBs,
               v1[i].bandwidth_GBs,
               v2[i].bandwidth_GBs,
               thr[i].bandwidth_GBs);
    }

    // ---- Table 3: % of T4 Peak Bandwidth ----
    printf("\n[3] %% OF T4 PEAK BANDWIDTH (320 GB/s)\n");
    printf("%-12s %12s %12s %12s\n",
           "N", "V1 Interl.", "V2 Sequen.", "Thrust");
    printf("%-12s %12s %12s %12s\n",
           "----------", "----------", "----------", "------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-12d %11.1f%% %11.1f%% %11.1f%%\n",
               Ns[i],
               v1[i].pct_peak_bw,
               v2[i].pct_peak_bw,
               thr[i].pct_peak_bw);
    }

    // ---- Table 4: Speedup over CPU ----
    printf("\n[4] SPEEDUP OVER CPU BASELINE\n");
    printf("%-12s %12s %12s %12s\n",
           "N", "V1 Interl.", "V2 Sequen.", "Thrust");
    printf("%-12s %12s %12s %12s\n",
           "----------", "----------", "----------", "------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-12d %11.1fx %11.1fx %11.1fx\n",
               Ns[i],
               cpu[i].time_ms / v1[i].time_ms,
               cpu[i].time_ms / v2[i].time_ms,
               cpu[i].time_ms / thr[i].time_ms);
    }

    // ---- Table 5: V2 speedup over V1 ----
    printf("\n[5] V2 SPEEDUP OVER V1\n");
    printf("%-12s %12s\n", "N", "V2 / V1");
    printf("%-12s %12s\n", "----------", "-------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-12d %11.2fx\n",
               Ns[i],
               v1[i].time_ms / v2[i].time_ms);
    }

    // ---- Table 6: V1 and V2 as % of Thrust ----
    printf("\n[6] BANDWIDTH AS %% OF THRUST\n");
    printf("%-12s %12s %12s\n",
           "N", "V1 / Thrust", "V2 / Thrust");
    printf("%-12s %12s %12s\n",
           "----------", "-----------", "-----------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-12d %11.1f%% %11.1f%%\n",
               Ns[i],
               v1[i].bandwidth_GBs / thr[i].bandwidth_GBs * 100.0f,
               v2[i].bandwidth_GBs / thr[i].bandwidth_GBs * 100.0f);
    }

    printf("\nNote: CPU bandwidth is not comparable to T4 peak.\n");
    printf("      GPU bandwidth = N * sizeof(float) / kernel_time\n");
    printf("      (reads N elements, writes 1 scalar — read-dominated)\n");
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {

    int default_sizes[] = {
        1 << 20,   //  1M  —   4.2 MB
        1 << 22,   //  4M  —  16.8 MB
        1 << 24,   // 16M  —  67.1 MB
        1 << 26,   // 64M  — 268.4 MB
    };
    int num_default = 4;

    int* sizes;
    int  num_sizes;
    int  single_size;
    if (argc > 1) {
        single_size = atoi(argv[1]);
        sizes       = &single_size;
        num_sizes   = 1;
    } else {
        sizes     = default_sizes;
        num_sizes = num_default;
    }

    // Header
    printf("============================================================\n");
    printf("HPS GPU Lab -- Experiment 3: Parallel Reduction\n");
    printf("Versions: CPU | V1 Interleaved | V2 Sequential | Thrust\n");
    printf("Block size: %d  |  Warmup: %d  |  Timed: %d\n",
           BLOCK_SIZE, WARMUP_RUNS, TIMED_RUNS);
    printf("============================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  |  CC %d.%d  |  %d SMs\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Peak BW: %.0f GB/s  |  Peak FP32: %.0f GFLOPS\n\n",
           (float)(2.0 * prop.memoryClockRate *
                   (prop.memoryBusWidth / 8) / 1e6),
           (float)(2.0 * prop.multiProcessorCount * 64 *
                   (prop.clockRate / 1e6)));

    BenchResult* cpu_res = new BenchResult[num_sizes];
    BenchResult* v1_res  = new BenchResult[num_sizes];
    BenchResult* v2_res  = new BenchResult[num_sizes];
    BenchResult* thr_res = new BenchResult[num_sizes];

    for (int i = 0; i < num_sizes; i++) {
        int    N     = sizes[i];
        size_t bytes = (size_t)N * sizeof(float);

        printf("------------------------------------------------------------\n");
        printf("N = %d  (%.1f MB)\n", N, bytes / 1e6f);
        printf("------------------------------------------------------------\n");

        // Host array — values in [1.0, 1.007], same pattern as matmul.cu
        float* h_A = (float*)malloc(bytes);
        for (int j = 0; j < N; j++)
            h_A[j] = 1.0f + (j & 7) * 0.001f;

        // CPU baseline + reference result for correctness checks
        printf("  CPU...            ");
        fflush(stdout);
        cpu_res[i]    = benchmark_cpu(h_A, N);
        float cpu_sum = reduce_cpu(h_A, N);
        printf("%.3f ms  (%.1f GB/s)  result=%.2f\n",
               cpu_res[i].time_ms, cpu_res[i].bandwidth_GBs, cpu_sum);

        // Upload to device once — all GPU versions share same input
        float* d_A;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

        // ---- V1: Interleaved addressing ----
        printf("  V1 Interleaved... ");
        fflush(stdout);
        v1_res[i] = benchmark_gpu(d_A, N, reduce_v1_interleaved);
        {
            int    grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            float* d_p;
            CUDA_CHECK(cudaMalloc(&d_p, grid * sizeof(float)));
            reduce_v1_interleaved<<<grid, BLOCK_SIZE,
                                    BLOCK_SIZE * sizeof(float)>>>(d_A, d_p, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            bool ok = verify(cpu_sum, reduce_partial_sums_on_cpu(d_p, grid), N);
            printf("%.4f ms  (%.1f GB/s, %.1f%% peak)  [%s]\n",
                   v1_res[i].time_ms, v1_res[i].bandwidth_GBs,
                   v1_res[i].pct_peak_bw, ok ? "OK" : "FAIL");
            CUDA_CHECK(cudaFree(d_p));
        }

        // ---- V2: Sequential addressing ----
        printf("  V2 Sequential...  ");
        fflush(stdout);
        v2_res[i] = benchmark_gpu(d_A, N, reduce_v2_sequential);
        {
            int    grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            float* d_p;
            CUDA_CHECK(cudaMalloc(&d_p, grid * sizeof(float)));
            reduce_v2_sequential<<<grid, BLOCK_SIZE,
                                   BLOCK_SIZE * sizeof(float)>>>(d_A, d_p, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            bool ok = verify(cpu_sum, reduce_partial_sums_on_cpu(d_p, grid), N);
            printf("%.4f ms  (%.1f GB/s, %.1f%% peak)  [%s]\n",
                   v2_res[i].time_ms, v2_res[i].bandwidth_GBs,
                   v2_res[i].pct_peak_bw, ok ? "OK" : "FAIL");
            CUDA_CHECK(cudaFree(d_p));
        }

        // ---- Thrust reference ----
        printf("  Thrust...         ");
        fflush(stdout);
        thr_res[i] = benchmark_thrust(d_A, N);
        {
            thrust::device_ptr<const float> ptr(d_A);
            float thrust_sum = thrust::reduce(ptr, ptr + N,
                                              0.0f, thrust::plus<float>());
            bool ok = verify(cpu_sum, thrust_sum, N);
            printf("%.4f ms  (%.1f GB/s, %.1f%% peak)  [%s]\n",
                   thr_res[i].time_ms, thr_res[i].bandwidth_GBs,
                   thr_res[i].pct_peak_bw, ok ? "OK" : "FAIL");
        }

        CUDA_CHECK(cudaFree(d_A));
        free(h_A);
    }

    print_summary(cpu_res, v1_res, v2_res, thr_res, sizes, num_sizes);

    delete[] cpu_res;
    delete[] v1_res;
    delete[] v2_res;
    delete[] thr_res;

    return 0;
}
