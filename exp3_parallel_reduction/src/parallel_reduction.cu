// ============================================================
// Experiment 3: Parallel Reduction
// hps_gpu_cuda_lab — github.com/high-perf-systems
//
// Versions implemented in this file:
//   1. CPU baseline   (sequential sum — serial dependency chain)
//   2. GPU V1         (interleaved addressing — warp divergence present)
//   3. GPU V2         (sequential addressing — divergence eliminated)
//   4. GPU V3         (first add during load — half the blocks, 1 fewer barrier)
//   5. GPU V4         (unroll last warp — remove last 5 barriers)
//   6. thrust::reduce (library reference baseline)
//
// Versions to be added:
//   7. GPU V5         (warp shuffle — no shared memory for last warp)
//
// Build:
//   nvcc -O2 -o parallel_sum parallel_sum.cu -lm
//
// Run all sizes:    ./parallel_sum
// Run single size:  ./parallel_sum 16777216
//
// Profile hardware metrics (ncu required on CC 7.5+):
//   ncu --section SpeedOfLight --section WarpStateStats \
//       --section InstructionStats --section MemoryWorkloadAnalysis \
//       --section Occupancy --kernel-name reduce_v4_unroll_last_warp \
//       ./parallel_sum 1048576 2>&1 | head -200
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
#define WARMUP_RUNS      3
#define TIMED_RUNS      10

#define T4_PEAK_BW_GBS   320.0f
#define T4_PEAK_GFLOPS  8141.0f

// ============================================================
// RESULT STRUCT
// ============================================================
typedef struct {
    int   N;
    float time_ms;
    float bandwidth_GBs;
    float pct_peak_bw;
} BenchResult;

// ============================================================
// GPU TIMER
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
// CPU TIMER
// ============================================================
static inline double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================
// BANDWIDTH HELPER
// ============================================================
static inline float compute_bw(int N, float time_ms) {
    double bytes   = (double)N * sizeof(float);
    double seconds = time_ms / 1000.0;
    return (float)(bytes / seconds / 1e9);
}

// ============================================================
// CORRECTNESS CHECK
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
// ============================================================
float reduce_cpu(const float* A, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        sum += A[i];
    return sum;
}

// ============================================================
// GPU V1 — INTERLEAVED ADDRESSING (warp divergence present)
// ============================================================
__global__ void reduce_v1_interleaved(
    const float* __restrict__ A,
    float*       __restrict__ partial_sums,
    int N)
{
    extern __shared__ float sdata[];
    int tid        = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (global_idx < N) ? A[global_idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

// ============================================================
// GPU V2 — SEQUENTIAL ADDRESSING (divergence eliminated first 3 steps)
// ============================================================
__global__ void reduce_v2_sequential(
    const float* __restrict__ A,
    float*       __restrict__ partial_sums,
    int N)
{
    extern __shared__ float sdata[];
    int tid        = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (global_idx < N) ? A[global_idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

// ============================================================
// GPU V3 — SEQUENTIAL ADDRESSING + FIRST ADD DURING LOAD
// Each thread loads two elements (blockDim.x apart) and adds them
// before the reduction loop. Half the blocks, one fewer barrier.
// ============================================================
__global__ void reduce_v3_first_add(
    const float* __restrict__ A,
    float*       __restrict__ partial_sums,
    int N)
{
    extern __shared__ float sdata[];
    int   tid       = threadIdx.x;
    int   global_id = blockIdx.x * (blockDim.x * 2) + tid;
    float sum       = 0.0f;

    if (global_id < N)              sum += A[global_id];
    if (global_id + blockDim.x < N) sum += A[global_id + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0];
}

// ============================================================
// WARP-LEVEL REDUCTION (used by V4)
//
// Once s < 32 only a single warp (threads 0-31) is active. Threads
// within a warp execute in lockstep (SIMT), so they are implicitly
// synchronised at the instruction level — no __syncthreads() needed.
//
// volatile is MANDATORY: it forces every sdata[] access to be a real
// shared-memory read/write. Without it, the compiler may cache values
// in registers across the 6 lines, so a thread would read a stale
// value instead of what its neighbour just wrote. lockstep gives
// ordering; volatile gives visibility — both are required.
//
// All 32 threads execute all 6 lines (no inner if-guard). Threads
// >= 16 compute junk into slots nobody reads, but that wasted work
// is cheaper than a branch (which would reintroduce divergence).
//
// NOTE: starting at sdata[tid + 32] assumes the caller's loop exits
// with exactly 32 active threads — true for BLOCK_SIZE = 256.
// ============================================================
__device__ void warpReduce(volatile float* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

// ============================================================
// GPU V4 — V3 + UNROLL LAST WARP (remove last 5 barriers)
//
// Identical to V3 through the load phase. The reduction loop now
// stops at s = 32 (condition s > 32). The final 5 steps (s = 32,
// 16, 8, 4, 2, 1) are handled by warpReduce() with NO barriers,
// since a single warp executes them in lockstep.
//
// Removes 5 of the block-wide __syncthreads() barriers per block.
// This also eliminates the residual divergence of the last 5 steps:
// warpReduce has no if-guard, so no warp is split active/idle.
//
// Remaining barriers: just the first 2-3 steps (s = 128, 64) plus
// the load barrier. Warp Cycles Per Issued Instruction should drop;
// predicated-off threads should fall toward zero.
// ============================================================
__global__ void reduce_v4_unroll_last_warp(
    const float* __restrict__ A,
    float*       __restrict__ partial_sums,
    int N)
{
    extern __shared__ float sdata[];
    int   tid       = threadIdx.x;
    int   global_id = blockIdx.x * (blockDim.x * 2) + tid;
    float sum       = 0.0f;

    if (global_id < N)              sum += A[global_id];
    if (global_id + blockDim.x < N) sum += A[global_id + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction down to the last warp (stop at s = 32)
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Last 5 steps — single warp, no barriers needed
    if (tid < 32)
        warpReduce(sdata, tid);

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
    br.N = N; br.time_ms = avg_ms;
    br.bandwidth_GBs = compute_bw(N, avg_ms);
    br.pct_peak_bw   = 0.0f;
    return br;
}

// ============================================================
// BENCHMARK: GPU KERNEL — generic helper
// elems_per_block = BLOCK_SIZE      → V1, V2 (one element / thread)
// elems_per_block = BLOCK_SIZE * 2  → V3, V4 (two elements / thread)
// grid_size = ceil(N / elems_per_block)
// ============================================================
typedef void (*ReduceKernel)(const float*, float*, int);

BenchResult benchmark_gpu(
    const float*  d_A,
    int           N,
    ReduceKernel  kernel,
    int           elems_per_block)
{
    int   grid_size  = (N + elems_per_block - 1) / elems_per_block;
    int   smem_bytes = BLOCK_SIZE * sizeof(float);
    float result     = 0.0f;

    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

    GPUTimer timer;

    for (int r = 0; r < WARMUP_RUNS; r++) {
        kernel<<<grid_size, BLOCK_SIZE, smem_bytes>>>(d_A, d_partial, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        result = reduce_partial_sums_on_cpu(d_partial, grid_size);
    }

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
    br.N = N; br.time_ms = avg_ms;
    br.bandwidth_GBs = compute_bw(N, avg_ms);
    br.pct_peak_bw   = br.bandwidth_GBs / T4_PEAK_BW_GBS * 100.0f;
    return br;
}

// ============================================================
// CORRECTNESS HELPER — run a kernel once, return its result
// ============================================================
float run_once_get_result(
    const float* d_A,
    int          N,
    ReduceKernel kernel,
    int          elems_per_block)
{
    int    grid = (N + elems_per_block - 1) / elems_per_block;
    float* d_p;
    CUDA_CHECK(cudaMalloc(&d_p, grid * sizeof(float)));
    kernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_A, d_p, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    float result = reduce_partial_sums_on_cpu(d_p, grid);
    CUDA_CHECK(cudaFree(d_p));
    return result;
}

// ============================================================
// BENCHMARK: THRUST
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
    br.N = N; br.time_ms = avg_ms;
    br.bandwidth_GBs = compute_bw(N, avg_ms);
    br.pct_peak_bw   = br.bandwidth_GBs / T4_PEAK_BW_GBS * 100.0f;
    return br;
}

// ============================================================
// PRINT SUMMARY TABLES
// ============================================================
void print_summary(
    const BenchResult* cpu,
    const BenchResult* v1,
    const BenchResult* v2,
    const BenchResult* v3,
    const BenchResult* v4,
    const BenchResult* thr,
    const int*         Ns,
    int                num_sizes)
{
    printf("\n");
    printf("========================================================================================\n");
    printf("SUMMARY — %d timed runs (after %d warmup)   |   T4 Peak BW: %.0f GB/s\n",
           TIMED_RUNS, WARMUP_RUNS, T4_PEAK_BW_GBS);
    printf("========================================================================================\n");

    // ---- Table 1: Time (ms) ----
    printf("\n[1] KERNEL TIME (ms)\n");
    printf("%-11s %8s %10s %10s %10s %10s %10s\n",
           "N", "CPU", "V1", "V2", "V3", "V4", "Thrust");
    printf("%-11s %8s %10s %10s %10s %10s %10s\n",
           "----------", "------", "--------", "--------", "--------", "--------", "------");
    for (int i = 0; i < num_sizes; i++)
        printf("%-11d %8.3f %10.4f %10.4f %10.4f %10.4f %10.4f\n",
               Ns[i], cpu[i].time_ms, v1[i].time_ms, v2[i].time_ms,
               v3[i].time_ms, v4[i].time_ms, thr[i].time_ms);

    // ---- Table 2: Bandwidth (GB/s) ----
    printf("\n[2] EFFECTIVE BANDWIDTH (GB/s)  —  T4 Peak: %.0f GB/s\n", T4_PEAK_BW_GBS);
    printf("%-11s %8s %10s %10s %10s %10s %10s\n",
           "N", "CPU", "V1", "V2", "V3", "V4", "Thrust");
    printf("%-11s %8s %10s %10s %10s %10s %10s\n",
           "----------", "------", "--------", "--------", "--------", "--------", "------");
    for (int i = 0; i < num_sizes; i++)
        printf("%-11d %8.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n",
               Ns[i], cpu[i].bandwidth_GBs, v1[i].bandwidth_GBs, v2[i].bandwidth_GBs,
               v3[i].bandwidth_GBs, v4[i].bandwidth_GBs, thr[i].bandwidth_GBs);

    // ---- Table 3: % of T4 Peak Bandwidth ----
    printf("\n[3] %% OF T4 PEAK BANDWIDTH (320 GB/s)\n");
    printf("%-11s %10s %10s %10s %10s %10s\n",
           "N", "V1", "V2", "V3", "V4", "Thrust");
    printf("%-11s %10s %10s %10s %10s %10s\n",
           "----------", "--------", "--------", "--------", "--------", "------");
    for (int i = 0; i < num_sizes; i++)
        printf("%-11d %9.1f%% %9.1f%% %9.1f%% %9.1f%% %9.1f%%\n",
               Ns[i], v1[i].pct_peak_bw, v2[i].pct_peak_bw,
               v3[i].pct_peak_bw, v4[i].pct_peak_bw, thr[i].pct_peak_bw);

    // ---- Table 4: Speedup over CPU ----
    printf("\n[4] SPEEDUP OVER CPU BASELINE\n");
    printf("%-11s %10s %10s %10s %10s %10s\n",
           "N", "V1", "V2", "V3", "V4", "Thrust");
    printf("%-11s %10s %10s %10s %10s %10s\n",
           "----------", "--------", "--------", "--------", "--------", "------");
    for (int i = 0; i < num_sizes; i++)
        printf("%-11d %9.1fx %9.1fx %9.1fx %9.1fx %9.1fx\n",
               Ns[i],
               cpu[i].time_ms / v1[i].time_ms,
               cpu[i].time_ms / v2[i].time_ms,
               cpu[i].time_ms / v3[i].time_ms,
               cpu[i].time_ms / v4[i].time_ms,
               cpu[i].time_ms / thr[i].time_ms);

    // ---- Table 5: Incremental speedups ----
    printf("\n[5] INCREMENTAL SPEEDUPS (each version vs the previous)\n");
    printf("%-11s %10s %10s %10s\n", "N", "V2/V1", "V3/V2", "V4/V3");
    printf("%-11s %10s %10s %10s\n", "----------", "--------", "--------", "--------");
    for (int i = 0; i < num_sizes; i++)
        printf("%-11d %9.2fx %9.2fx %9.2fx\n",
               Ns[i],
               v1[i].time_ms / v2[i].time_ms,
               v2[i].time_ms / v3[i].time_ms,
               v3[i].time_ms / v4[i].time_ms);

    // ---- Table 6: Bandwidth as % of Thrust ----
    printf("\n[6] BANDWIDTH AS %% OF THRUST\n");
    printf("%-11s %10s %10s %10s %10s\n",
           "N", "V1", "V2", "V3", "V4");
    printf("%-11s %10s %10s %10s %10s\n",
           "----------", "--------", "--------", "--------", "--------");
    for (int i = 0; i < num_sizes; i++)
        printf("%-11d %9.1f%% %9.1f%% %9.1f%% %9.1f%%\n",
               Ns[i],
               v1[i].bandwidth_GBs / thr[i].bandwidth_GBs * 100.0f,
               v2[i].bandwidth_GBs / thr[i].bandwidth_GBs * 100.0f,
               v3[i].bandwidth_GBs / thr[i].bandwidth_GBs * 100.0f,
               v4[i].bandwidth_GBs / thr[i].bandwidth_GBs * 100.0f);

    printf("\nNote: GPU bandwidth = N * sizeof(float) / kernel_time\n");
    printf("      (reads N elements, writes 1 scalar — read-dominated)\n");
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {

    int default_sizes[] = { 1<<20, 1<<22, 1<<24, 1<<26 };
    int num_default = 4;

    int* sizes; int num_sizes; int single_size;
    if (argc > 1) {
        single_size = atoi(argv[1]);
        sizes = &single_size; num_sizes = 1;
    } else {
        sizes = default_sizes; num_sizes = num_default;
    }

    printf("============================================================\n");
    printf("HPS GPU Lab -- Experiment 3: Parallel Reduction\n");
    printf("Versions: CPU | V1 | V2 | V3 | V4 | Thrust\n");
    printf("Block size: %d  |  Warmup: %d  |  Timed: %d\n",
           BLOCK_SIZE, WARMUP_RUNS, TIMED_RUNS);
    printf("============================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  |  CC %d.%d  |  %d SMs\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Peak BW: %.0f GB/s  |  Peak FP32: %.0f GFLOPS\n\n",
           (float)(2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6),
           (float)(2.0 * prop.multiProcessorCount * 64 * (prop.clockRate / 1e6)));

    BenchResult* cpu_res = new BenchResult[num_sizes];
    BenchResult* v1_res  = new BenchResult[num_sizes];
    BenchResult* v2_res  = new BenchResult[num_sizes];
    BenchResult* v3_res  = new BenchResult[num_sizes];
    BenchResult* v4_res  = new BenchResult[num_sizes];
    BenchResult* thr_res = new BenchResult[num_sizes];

    for (int i = 0; i < num_sizes; i++) {
        int    N     = sizes[i];
        size_t bytes = (size_t)N * sizeof(float);

        printf("------------------------------------------------------------\n");
        printf("N = %d  (%.1f MB)\n", N, bytes / 1e6f);
        printf("------------------------------------------------------------\n");

        float* h_A = (float*)malloc(bytes);
        for (int j = 0; j < N; j++)
            h_A[j] = 1.0f + (j & 7) * 0.001f;

        printf("  CPU...            "); fflush(stdout);
        cpu_res[i]    = benchmark_cpu(h_A, N);
        float cpu_sum = reduce_cpu(h_A, N);
        printf("%.3f ms  (%.1f GB/s)  result=%.2f\n",
               cpu_res[i].time_ms, cpu_res[i].bandwidth_GBs, cpu_sum);

        float* d_A;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

        // ---- V1 ----
        printf("  V1 Interleaved... "); fflush(stdout);
        v1_res[i] = benchmark_gpu(d_A, N, reduce_v1_interleaved, BLOCK_SIZE);
        {
            float gpu = run_once_get_result(d_A, N, reduce_v1_interleaved, BLOCK_SIZE);
            bool  ok  = verify(cpu_sum, gpu, N);
            printf("%.4f ms  (%.1f GB/s, %.1f%% peak)  [%s]\n",
                   v1_res[i].time_ms, v1_res[i].bandwidth_GBs, v1_res[i].pct_peak_bw, ok ? "OK" : "FAIL");
        }

        // ---- V2 ----
        printf("  V2 Sequential...  "); fflush(stdout);
        v2_res[i] = benchmark_gpu(d_A, N, reduce_v2_sequential, BLOCK_SIZE);
        {
            float gpu = run_once_get_result(d_A, N, reduce_v2_sequential, BLOCK_SIZE);
            bool  ok  = verify(cpu_sum, gpu, N);
            printf("%.4f ms  (%.1f GB/s, %.1f%% peak)  [%s]\n",
                   v2_res[i].time_ms, v2_res[i].bandwidth_GBs, v2_res[i].pct_peak_bw, ok ? "OK" : "FAIL");
        }

        // ---- V3 (2 elems/thread → half blocks) ----
        printf("  V3 AddLoad...     "); fflush(stdout);
        v3_res[i] = benchmark_gpu(d_A, N, reduce_v3_first_add, BLOCK_SIZE * 2);
        {
            float gpu = run_once_get_result(d_A, N, reduce_v3_first_add, BLOCK_SIZE * 2);
            bool  ok  = verify(cpu_sum, gpu, N);
            printf("%.4f ms  (%.1f GB/s, %.1f%% peak)  [%s]\n",
                   v3_res[i].time_ms, v3_res[i].bandwidth_GBs, v3_res[i].pct_peak_bw, ok ? "OK" : "FAIL");
        }

        // ---- V4 (unroll last warp, also 2 elems/thread → half blocks) ----
        printf("  V4 UnrollWarp...  "); fflush(stdout);
        v4_res[i] = benchmark_gpu(d_A, N, reduce_v4_unroll_last_warp, BLOCK_SIZE * 2);
        {
            float gpu = run_once_get_result(d_A, N, reduce_v4_unroll_last_warp, BLOCK_SIZE * 2);
            bool  ok  = verify(cpu_sum, gpu, N);
            printf("%.4f ms  (%.1f GB/s, %.1f%% peak)  [%s]\n",
                   v4_res[i].time_ms, v4_res[i].bandwidth_GBs, v4_res[i].pct_peak_bw, ok ? "OK" : "FAIL");
        }

        // ---- Thrust ----
        printf("  Thrust...         "); fflush(stdout);
        thr_res[i] = benchmark_thrust(d_A, N);
        {
            thrust::device_ptr<const float> ptr(d_A);
            float thrust_sum = thrust::reduce(ptr, ptr + N, 0.0f, thrust::plus<float>());
            bool ok = verify(cpu_sum, thrust_sum, N);
            printf("%.4f ms  (%.1f GB/s, %.1f%% peak)  [%s]\n",
                   thr_res[i].time_ms, thr_res[i].bandwidth_GBs, thr_res[i].pct_peak_bw, ok ? "OK" : "FAIL");
        }

        CUDA_CHECK(cudaFree(d_A));
        free(h_A);
    }

    print_summary(cpu_res, v1_res, v2_res, v3_res, v4_res, thr_res, sizes, num_sizes);

    delete[] cpu_res; delete[] v1_res; delete[] v2_res;
    delete[] v3_res;  delete[] v4_res; delete[] thr_res;
    return 0;
}
