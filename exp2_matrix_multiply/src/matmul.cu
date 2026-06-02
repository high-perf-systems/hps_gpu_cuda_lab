// ============================================================
// Experiment 2: Matrix Multiplication
// hps_gpu_cuda_lab — github.com/high-perf-systems
//
// Four versions:
//   1. CPU baseline (sequential)
//   2. GPU Naive   (global memory only)
//   3. GPU Tiled   (shared memory, TILE=16)
//   4. GPU Tiled   (shared memory, TILE=32)
//
// Build:
//   nvcc -O2 -o matmul matmul.cu -lm
//
// Run all sizes:
//   ./matmul
//
// Run single size:
//   ./matmul 1024
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// ERROR CHECKING
#define CUDA_CHECK(call)                                \
do                                                      \
{                                                       \  
    cudaError_t err = (call);                           \
    if (err != cudaSuccess){                            \
        fprintf(stderr, "CUDA error at %s:%d -- %s\n",  \
        __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while (0) 

// CONFIGS
#define WARMUP_RUNS 3
#define TIMED_RUNS 10

#define TILE_16 16
#define TILE_32 32

// RESULT STRUCT
typedef struct{
    int N;
    float time_ms; // average kernel compute time
    float gflops; // Billion floating point operations per second
    float bandwidth_GBs;  // effective memory bandwidth
} BenchResult;

// GPU timer
struct GPUTimer
{
    cudaEvent_t start, stop;
    GPUTimer(){
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GPUTimer()
    {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    void Start()
    {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }
    void Stop()
    {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
    }
    float ElapsedMs(){
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }

};

// CPU timer
static inline double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Performance helpers
// glfops
float compute_gflops(int N, float time_ms)
{
    double flops = 2 * (double)N * N * N;
    double seconds = time_ms / 1000.0;
    return (float) (flops / seconds / 1e9);
}

// minimum bandwidth
float compute_bandwidth(int N, float time_ms) {
    double bytes   = 3.0 * (double)N * N * sizeof(float);
    double seconds = time_ms / 1000.0;
    return (float)(bytes / seconds / 1e9);
}

// CORRECTNESS CHECKS
bool verify(const float* ref,
            const float* result,
            int N)
{
    // Epsilon scales with N because we sum N products
    // Each product has ~1e-7 relative error (float precision)
    // After N additions: accumulated error ~ N * 1e-7
    float eps = (float)N * 1e-4f;

    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(ref[i] - result[i]);
        float mag  = fabsf(ref[i]) + 1e-6f;
        if (diff / mag > eps) {
            printf("  MISMATCH at [%d]: ref=%.4f got=%.4f\n",
                   i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}

// VERSION 1 : CPU BASELINE
// Assumption : Each of A, B, C are N*N square matrices
void matmul_cpu(
    const float* A,
    const float* B,
    float* C, 
    int N
)
{
    for (int i=0;i<N;i++)
    {
        for (int j=0;j<N;j++)
        {
            float sum = 0.0f;
            for (int k=0;k<N;k++)
            {
                sum += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = sum;

        }
    }
}

// VERSION 2 : GPU NAIVE
// Same set of matrices A, B, C
// one thread per output element with data loaded from global memory
__global__ void matmul_naive_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    // 2D thread indexing
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;
    float sum = 0.0f;
    // A read access is sequential, while B read access is strided
    for (int k = 0; k < N;k++)
    {
        sum += A[row*N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// VERSION 3 : GPU TILED
// Same set of matrices A, B, C
// Division of A and B into TILE*TILE tiles and loaded into shared memory
template <int TILE>
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C,
    int N)
    {
        // __shared__ costs just 5 cycles
        __shared__ float tile_A[TILE][TILE];
        __shared__ float tile_B[TILE][TILE];
        // THREAD -> outptu element mapping
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        int row = blockIdx.y * blockDim.y + ty; // global row index
        int col = blockIdx.x * blockDim.x + tx; // global col index

        // private accumulator
        float sum = 0.0f;
        int num_tiles = (N + TILE - 1) / TILE;
        for (int t = 0; t < num_tiles; t++)
        {
            int a_col = t * TILE + tx;
            int b_row = t * TILE + ty;

            tile_A[ty][tx] = (row < N && a_col < N) ?
                                A[row * N + a_col] : 0.0f;
            tile_B[ty][tx] = (col < N && b_row < N) ?
                                B[b_row * N + col] : 0.0f;
            // sync all the threads in the block
            __syncthreads();
            // compute matmul in these tiles
            #pragma unroll
            for(int k=0;k<TILE;k++)
            {
                sum += tile_A[ty][k] * tile_B[k][tx];
            }
            // sync threads needed again 
            __syncthreads();
        }
        // by the time the sum for this row and col would have been 
       // accumulated by all the blocks
       if (row < N && col < N)
        C[row * N + col] = sum;
    }

// ============================================================
// BENCHMARK: CPU
// ============================================================
BenchResult benchmark_cpu(const float* A,
                           const float* B,
                           float* C,
                           int N)
{
    // Warmup
    for (int r = 0; r < WARMUP_RUNS; r++)
        matmul_cpu(A, B, C, N);

    // Timed runs
    double total = 0.0;
    for (int r = 0; r < TIMED_RUNS; r++) {
        double t0 = now_ms();
        matmul_cpu(A, B, C, N);
        total += now_ms() - t0;
    }

    float ms = (float)(total / TIMED_RUNS);

    BenchResult res;
    res.N            = N;
    res.time_ms      = ms;
    res.gflops       = compute_gflops(N, ms);
    res.bandwidth_GBs = compute_bandwidth(N, ms);
    return res;
}

// ============================================================
// BENCHMARK: GPU (works for all three GPU versions)
// ============================================================
// Takes a function pointer approach via kernel_id:
//   0 = naive
//   1 = tiled T=16
//   2 = tiled T=32
// ============================================================
BenchResult benchmark_gpu(const float* h_A,
                           const float* h_B,
                           float* h_C,
                           int N,
                           int kernel_id)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Upload A and B once (same data for all runs)
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes,
                          cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // GRID AND BLOCK DIMENSIONS
    // --------------------------------------------------------
    // We use 2D thread blocks matching the tile size.
    // Grid covers the entire N×N output matrix.
    //
    // For TILE=16: block = (16, 16) = 256 threads
    // For TILE=32: block = (32, 32) = 1024 threads (T4 max!)
    //
    // Grid = ceil(N/TILE) × ceil(N/TILE) blocks
    // --------------------------------------------------------
    dim3 block_naive(16, 16);  // 256 threads (arbitrary for naive)
    dim3 grid_naive(
        (N + 15) / 16,
        (N + 15) / 16);

    dim3 block_t16(TILE_16, TILE_16);  // 16×16 = 256 threads
    dim3 grid_t16(
        (N + TILE_16 - 1) / TILE_16,
        (N + TILE_16 - 1) / TILE_16);

    dim3 block_t32(TILE_32, TILE_32);  // 32×32 = 1024 threads
    dim3 grid_t32(
        (N + TILE_32 - 1) / TILE_32,
        (N + TILE_32 - 1) / TILE_32);

    GPUTimer timer;

    // --------------------------------------------------------
    // WARMUP RUNS
    // --------------------------------------------------------
    for (int r = 0; r < WARMUP_RUNS; r++) {
        if (kernel_id == 0)
            matmul_naive_kernel<<<grid_naive, block_naive>>>(
                d_A, d_B, d_C, N);
        else if (kernel_id == 1)
            matmul_tiled_kernel<TILE_16><<<grid_t16, block_t16>>>(
                d_A, d_B, d_C, N);
        else
            matmul_tiled_kernel<TILE_32><<<grid_t32, block_t32>>>(
                d_A, d_B, d_C, N);

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --------------------------------------------------------
    // TIMED RUNS
    // --------------------------------------------------------
    float total_ms = 0.0f;
    for (int r = 0; r < TIMED_RUNS; r++) {
        timer.Start();

        if (kernel_id == 0)
            matmul_naive_kernel<<<grid_naive, block_naive>>>(
                d_A, d_B, d_C, N);
        else if (kernel_id == 1)
            matmul_tiled_kernel<TILE_16><<<grid_t16, block_t16>>>(
                d_A, d_B, d_C, N);
        else
            matmul_tiled_kernel<TILE_32><<<grid_t32, block_t32>>>(
                d_A, d_B, d_C, N);

        CUDA_CHECK(cudaGetLastError());
        timer.Stop();
        total_ms += timer.ElapsedMs();
    }

    // Copy result back for verification
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    float avg_ms = total_ms / TIMED_RUNS;

    BenchResult res;
    res.N             = N;
    res.time_ms       = avg_ms;
    res.gflops        = compute_gflops(N, avg_ms);
    res.bandwidth_GBs = compute_bandwidth(N, avg_ms);
    return res;
}

// ============================================================
// PRINT SUMMARY TABLE
// ============================================================
void print_summary(
    const BenchResult* cpu,
    const BenchResult* naive,
    const BenchResult* t16,
    const BenchResult* t32,
    const int*         sizes,
    int                num_sizes,
    float              peak_gflops,
    float              peak_bw)
{
    printf("\n");
    printf("================================================================"
           "=============\n");
    printf("SUMMARY — all N, averaged over %d runs (after %d warmup)\n",
           TIMED_RUNS, WARMUP_RUNS);
    printf("================================================================"
           "=============\n");

    // ---- Table 1: Time (ms) ----
    printf("\n[1] KERNEL TIME (ms)\n");
    printf("%-8s %12s %12s %14s %14s\n",
           "N", "CPU", "Naive GPU", "Tiled T=16", "Tiled T=32");
    printf("%-8s %12s %12s %14s %14s\n",
           "----", "---------", "---------", "----------", "----------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-8d %12.3f %12.4f %14.4f %14.4f\n",
               sizes[i],
               cpu[i].time_ms,
               naive[i].time_ms,
               t16[i].time_ms,
               t32[i].time_ms);
    }

    // ---- Table 2: GFLOPS ----
    printf("\n[2] PERFORMANCE (GFLOPS)  —  T4 Peak FP32: %.0f GFLOPS\n",
           peak_gflops);
    printf("%-8s %10s %12s %14s %14s %14s\n",
           "N", "CPU", "Naive GPU", "Tiled T=16",
           "Tiled T=32", "Peak%% (T32)");
    printf("%-8s %10s %12s %14s %14s %14s\n",
           "----", "---", "---------", "----------",
           "----------", "-----------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-8d %10.2f %12.2f %14.2f %14.2f %13.1f%%\n",
               sizes[i],
               cpu[i].gflops,
               naive[i].gflops,
               t16[i].gflops,
               t32[i].gflops,
               t32[i].gflops / peak_gflops * 100.0f);
    }

    // ---- Table 3: Speedups ----
    printf("\n[3] SPEEDUP OVER CPU BASELINE\n");
    printf("%-8s %12s %14s %14s\n",
           "N", "Naive GPU", "Tiled T=16", "Tiled T=32");
    printf("%-8s %12s %14s %14s\n",
           "----", "---------", "----------", "----------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-8d %11.1fx %13.1fx %13.1fx\n",
               sizes[i],
               cpu[i].time_ms / naive[i].time_ms,
               cpu[i].time_ms / t16[i].time_ms,
               cpu[i].time_ms / t32[i].time_ms);
    }

    // ---- Table 4: Tiled vs Naive Speedup ----
    printf("\n[4] TILED SPEEDUP OVER NAIVE GPU\n");
    printf("%-8s %14s %14s\n",
           "N", "T=16 vs Naive", "T=32 vs Naive");
    printf("%-8s %14s %14s\n",
           "----", "-------------", "-------------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-8d %13.1fx %13.1fx\n",
               sizes[i],
               naive[i].time_ms / t16[i].time_ms,
               naive[i].time_ms / t32[i].time_ms);
    }

    // ---- Table 5: Bandwidth ----
    printf("\n[5] EFFECTIVE BANDWIDTH (GB/s)"
           "  —  T4 Peak: %.0f GB/s\n", peak_bw);
    printf("%-8s %10s %12s %14s %14s\n",
           "N", "CPU", "Naive GPU", "Tiled T=16", "Tiled T=32");
    printf("%-8s %10s %12s %14s %14s\n",
           "----", "---", "---------", "----------", "----------");
    for (int i = 0; i < num_sizes; i++) {
        printf("%-8d %10.1f %12.1f %14.1f %14.1f\n",
               sizes[i],
               cpu[i].bandwidth_GBs,
               naive[i].bandwidth_GBs,
               t16[i].bandwidth_GBs,
               t32[i].bandwidth_GBs);
    }

    // Note on bandwidth interpretation
    printf("\n  Note: bandwidth here = 3*N²*4 / time\n");
    printf("  This is ARITHMETIC minimum bytes (read A,B + write C).\n");
    printf("  Naive GPU actually reads MORE than this due to\n");
    printf("  poor caching of B (strided access wastes cache lines).\n");
    printf("  Tiled GPU reads LESS per element via shared memory reuse.\n");
    printf("  So: high BW for naive = fast despite wasted reads\n");
    printf("      lower BW for tiled = fewer DRAM reads, more compute\n");
}

int main(int argc, char** argv)
{
    // --------------------------------------------------------
    // WHICH SIZES TO RUN?
    // --------------------------------------------------------
    // Default: 256, 512, 1024, 2048
    // Override: ./matmul 1024  (single size)
    //
    // Note: CPU becomes very slow at N=2048!
    // We cap CPU runs at N<=1024 to avoid waiting forever.
    // --------------------------------------------------------
    int default_sizes[] = {256, 512, 1024, 2048};
    int num_default =  4;
    int *sizes;
    int num_sizes;
    int single_size;
    if (argc > 1)
    {
        single_size = atoi(argv[1]);
        sizes = &single_size;
        num_sizes = 1;
    }
    else
    {
        sizes = default_sizes;
        num_sizes = num_default;
    }
        // --------------------------------------------------------
    // PRINT HEADER
    // --------------------------------------------------------
    printf("============================================================\n");
    printf("HPS GPU Lab -- Experiment 2: Matrix Multiplication\n");
    printf("Versions: CPU | Naive GPU | Tiled T=16 | Tiled T=32\n");
    printf("Warmup: %d | Timed: %d\n", WARMUP_RUNS, TIMED_RUNS);
    printf("============================================================\n");
    // GPU Info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6f;
    float peak_gflops = 2.0f * prop.multiProcessorCount * 64 * (prop.clockRate / 1e6f);
    printf("GPU: %s | CC %d.%d | %d SMs\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Peak BW: %.0f GB/s | Peak FP32: %.0f GFLOPS\n\n",
           peak_bw, peak_gflops);
    // allocate benchmark arrays
    BenchResult* cpu_res = new BenchResult[num_sizes];
    BenchResult* gpu_naive_res = new BenchResult[num_sizes];
    BenchResult* gpu_tiled16_res = new  BenchResult[num_sizes];
    BenchResult* gpu_tiled32_res = new BenchResult[num_sizes];

    // MAIN LOOP
    for (int i=0;i < num_sizes;i++)
    {
        int N = sizes[i];
              size_t bytes = (size_t)N * N * sizeof(float);

        printf("------------------------------------------------\n");
        printf("N = %d  (%dx%d matrices, %.1f MB each)\n",
               N, N, N, bytes / 1e6f);
        printf("------------------------------------------------\n");

        // Allocate host matrices
        float* h_A     = (float*)malloc(bytes);
        float* h_B     = (float*)malloc(bytes);
        float* h_C_ref = (float*)malloc(bytes); // CPU result
        float* h_C_gpu = (float*)malloc(bytes); // GPU result

        // Initialize with small values to avoid overflow
        // A[i][j] = small float, B[i][j] = small float
        // Use 1/N scaling so accumulated sums stay ~1.0
        for (int j = 0; j < N * N; j++) {
            h_A[j] = (float)(rand() % 10) / (float)N;
            h_B[j] = (float)(rand() % 10) / (float)N;
        }
                // ---- CPU (skip for N=2048 — too slow) ----
        if (N <= 1024) {
            printf("  CPU...        ");
            fflush(stdout);
            cpu_res[i] = benchmark_cpu(h_A, h_B, h_C_ref, N);
            printf("%.3f ms  (%.2f GFLOPS)\n",
                   cpu_res[i].time_ms, cpu_res[i].gflops);
        } else {
            printf("  CPU...        SKIPPED (N=%d too slow)\n", N);
            // Fill with sentinel so table prints cleanly
            cpu_res[i].N = N;
            cpu_res[i].time_ms = -1.0f;
            cpu_res[i].gflops  = -1.0f;
            cpu_res[i].bandwidth_GBs = -1.0f;
            // Run CPU once for reference (needed for verify)
            printf("  CPU ref...    ");
            fflush(stdout);
            double t0 = now_ms();
            matmul_cpu(h_A, h_B, h_C_ref, N);
            printf("%.3f ms (single run for verify)\n",
                   (float)(now_ms() - t0));
        }
                // ---- GPU Naive ----
        printf("  Naive GPU...  ");
        fflush(stdout);
        gpu_naive_res[i] = benchmark_gpu(h_A, h_B, h_C_gpu, N, 0);
        bool ok_naive = verify(h_C_ref, h_C_gpu, N);
        printf("%.4f ms  (%.1f GFLOPS)  [%s]\n",
               gpu_naive_res[i].time_ms, gpu_naive_res[i].gflops,
               ok_naive ? "OK" : "FAIL");

        // ---- GPU Tiled T=16 ----
        printf("  Tiled T=16... ");
        fflush(stdout);
        gpu_tiled16_res[i] = benchmark_gpu(h_A, h_B, h_C_gpu, N, 1);
        bool ok_t16 = verify(h_C_ref, h_C_gpu, N);
        printf("%.4f ms  (%.1f GFLOPS)  [%s]\n",
               gpu_tiled16_res[i].time_ms, gpu_tiled16_res[i].gflops,
               ok_t16 ? "OK" : "FAIL");

        // ---- GPU Tiled T=32 ----
        printf("  Tiled T=32... ");
        fflush(stdout);
        gpu_tiled32_res[i] = benchmark_gpu(h_A, h_B, h_C_gpu, N, 2);
        bool ok_t32 = verify(h_C_ref, h_C_gpu, N);
        printf("%.4f ms  (%.1f GFLOPS)  [%s]\n",
               gpu_tiled32_res[i].time_ms, gpu_tiled32_res[i].gflops,
               ok_t32 ? "OK" : "FAIL");

        free(h_A); free(h_B); free(h_C_ref); free(h_C_gpu);
    }
    // print the full summary
    print_summary(cpu_res, gpu_naive_res, gpu_tiled16_res, gpu_tiled32_res, sizes, num_sizes,
    peak_gflops, peak_bw);

    delete[] cpu_res;
    delete[] gpu_naive_res;
    delete[] gpu_tiled16_res;
    delete[] gpu_tiled32_res;

    return 0;
}
