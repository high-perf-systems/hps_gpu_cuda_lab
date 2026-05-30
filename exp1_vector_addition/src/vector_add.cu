// ============================================================
// Experiment 1: Vector Addition
// hps_gpu_cuda_lab — github.com/high-perf-systems
//
// Build:
//   nvcc -O2 -o vector_add vector_add.cu -lm
//
// Run (all N values at once):
//   ./vector_add
//
// Run (single specific N):
//   ./vector_add 22   → N = 1<<22 only
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr,                                       \
                "CUDA error at %s:%d -- %s\n",                   \
                __FILE__, __LINE__,                               \
                cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#define WARMUP_RUNS  3
#define TIMED_RUNS   10

// ============================================================
// RESULT STRUCT
// ============================================================
// Stores ALL timing data for one (N, memory_type) run.
// Returned from benchmark functions so main() can
// print a clean summary table across all N values.
// ============================================================
typedef struct {
    int   N;
    float cpu_ms;
    float h2d_ms;        // avg H2D transfer
    float kernel_ms;     // avg kernel
    float d2h_ms;        // avg D2H transfer
    float total_ms;      // h2d + kernel + d2h
    float bandwidth_GBs; // kernel bandwidth
} BenchResult;

// ============================================================
// GPU TIMER
// ============================================================
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    void Start() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }
    void Stop() {
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
// KERNEL
// ============================================================
__global__ void vector_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    C[idx] = A[idx] + B[idx];
}

// ============================================================
// CPU BASELINE
// ============================================================
void vector_add_cpu(const float* A,
                    const float* B,
                    float* C,
                    int N)
{
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// CORRECTNESS CHECK
// ============================================================
bool verify(const float* ref,
            const float* result,
            int N)
{
    const float eps = 1e-5f;
    for (int i = 0; i < N; i++) {
        if (fabsf(ref[i] - result[i]) > eps) {
            printf("  MISMATCH at [%d]: ref=%.4f got=%.4f\n",
                   i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}

// ============================================================
// BANDWIDTH
// ============================================================
float bandwidth_GBs(int N, float time_ms)
{
    double bytes   = (double)N * 3 * sizeof(float);
    double seconds = time_ms / 1000.0;
    return (float)(bytes / seconds / 1e9);
}

// ============================================================
// BENCHMARK: CPU
// ============================================================
float benchmark_cpu(const float* h_A,
                    const float* h_B,
                    float*       h_C,
                    int N)
{
    // Warmup
    for (int r = 0; r < WARMUP_RUNS; r++)
        vector_add_cpu(h_A, h_B, h_C, N);

    // Timed
    double total = 0.0;
    for (int r = 0; r < TIMED_RUNS; r++) {
        double t0 = now_ms();
        vector_add_cpu(h_A, h_B, h_C, N);
        total += now_ms() - t0;
    }
    return (float)(total / TIMED_RUNS);
}

// ============================================================
// BENCHMARK: GPU (generic — works for both pageable + pinned)
// ============================================================
// Takes pointers to host memory (caller decides pageable/pinned)
// Returns BenchResult with all timing fields filled.
//
// This is the KEY change: instead of printing inside the
// benchmark function, we RETURN the data.
// main() collects all results and prints ONE clean table.
// ============================================================
BenchResult benchmark_gpu(
    const float* h_A,
    const float* h_B,
    float*       h_C,
    int N,
    int threads_per_block,
    float cpu_ms)          // passed in for speedup calculation
{
    size_t bytes = (size_t)N * sizeof(float);
    int grid = (N + threads_per_block - 1) / threads_per_block;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    GpuTimer timer;

    // Warmup runs — discard timing
    for (int r = 0; r < WARMUP_RUNS; r++) {
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes,
                              cudaMemcpyHostToDevice));
        vector_add_kernel<<<grid, threads_per_block>>>(
            d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes,
                              cudaMemcpyDeviceToHost));
    }

    // Timed runs — accumulate
    float h2d_total    = 0.0f;
    float kernel_total = 0.0f;
    float d2h_total    = 0.0f;

    for (int r = 0; r < TIMED_RUNS; r++) {

        timer.Start();
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes,
                              cudaMemcpyHostToDevice));
        timer.Stop();
        h2d_total += timer.ElapsedMs();

        timer.Start();
        vector_add_kernel<<<grid, threads_per_block>>>(
            d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        timer.Stop();
        kernel_total += timer.ElapsedMs();

        timer.Start();
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes,
                              cudaMemcpyDeviceToHost));
        timer.Stop();
        d2h_total += timer.ElapsedMs();
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Pack all results into struct
    BenchResult r;
    r.N             = N;
    r.cpu_ms        = cpu_ms;
    r.h2d_ms        = h2d_total    / TIMED_RUNS;
    r.kernel_ms     = kernel_total / TIMED_RUNS;
    r.d2h_ms        = d2h_total    / TIMED_RUNS;
    r.total_ms      = r.h2d_ms + r.kernel_ms + r.d2h_ms;
    r.bandwidth_GBs = bandwidth_GBs(N, r.kernel_ms);
    return r;
}

// ============================================================
// PRINT DETAILED RESULTS FOR ONE N VALUE
// ============================================================
void print_detailed(int N,
                    float cpu_ms,
                    const BenchResult& pg,   // pageable
                    const BenchResult& pin,  // pinned
                    bool correct_pg,
                    bool correct_pin)
{
    printf("\n--------------------------------------------\n");
    printf("N = %d (1 << %d)  |  %.2f MB per array\n",
           N, (int)log2f((float)N),
           N * sizeof(float) / 1e6f);
    printf("--------------------------------------------\n");
    printf("%-18s %10s %10s %10s %10s %10s\n",
           "Version", "H2D(ms)", "Kernel(ms)",
           "D2H(ms)", "Total(ms)", "BW(GB/s)");
    printf("%-18s %10s %10s %10s %10s %10s\n",
           "-------", "-------", "----------",
           "-------", "---------", "--------");
    printf("%-18s %10s %10.4f %10s %10.4f %10s\n",
           "CPU", "-", cpu_ms, "-", cpu_ms, "-");
    printf("%-18s %10.4f %10.4f %10.4f %10.4f %10.1f  %s\n",
           "GPU Pageable",
           pg.h2d_ms, pg.kernel_ms, pg.d2h_ms,
           pg.total_ms, pg.bandwidth_GBs,
           correct_pg ? "[OK]" : "[FAIL]");
    printf("%-18s %10.4f %10.4f %10.4f %10.4f %10.1f  %s\n",
           "GPU Pinned",
           pin.h2d_ms, pin.kernel_ms, pin.d2h_ms,
           pin.total_ms, pin.bandwidth_GBs,
           correct_pin ? "[OK]" : "[FAIL]");
}

// ============================================================
// PRINT FINAL SUMMARY TABLE ACROSS ALL N VALUES
// ============================================================
// This is the KEY new function.
// Prints ONE clean table comparing all N values side by side.
//
// Shows:
//   - Kernel speedup (GPU vs CPU, compute only)
//   - Transfer speedup (pinned vs pageable)
//   - Bandwidth efficiency (% of T4 peak)
// ============================================================
void print_summary_table(
    const BenchResult* pg_results,    // pageable results array
    const BenchResult* pin_results,   // pinned results array
    const float*       cpu_results,   // cpu times array
    const int*         n_bits_arr,    // bit counts array
    int                num_sizes,     // how many N values
    float              peak_bw)       // GPU peak bandwidth
{
    printf("\n");
    printf("============================================================"
           "===================\n");
    printf("SUMMARY TABLE — all N values, averaged over %d runs "
           "(after %d warmup)\n",
           TIMED_RUNS, WARMUP_RUNS);
    printf("============================================================"
           "===================\n");

    // ---- Table 1: Kernel Performance ----
    printf("\n[1] KERNEL PERFORMANCE (compute only, excludes transfers)\n");
    printf("%-8s %10s %12s %12s %10s %10s\n",
           "N", "CPU(ms)", "GPU-Pg(ms)", "GPU-Pin(ms)",
           "Speedup-Pg", "Speedup-Pin");
    printf("%-8s %10s %12s %12s %10s %10s\n",
           "----", "-------", "----------", "----------",
           "----------", "-----------");
    for (int i = 0; i < num_sizes; i++) {
        int N = pg_results[i].N;
        printf("%-8d %10.4f %12.4f %12.4f %9.1fx %10.1fx\n",
               N,
               cpu_results[i],
               pg_results[i].kernel_ms,
               pin_results[i].kernel_ms,
               cpu_results[i] / pg_results[i].kernel_ms,
               cpu_results[i] / pin_results[i].kernel_ms);
    }

    // ---- Table 2: Transfer Performance ----
    printf("\n[2] TRANSFER PERFORMANCE (pinned vs pageable)\n");
    printf("%-8s %14s %14s %10s %14s %14s %10s\n",
           "N",
           "H2D-Pg(ms)", "H2D-Pin(ms)", "H2D Speedup",
           "D2H-Pg(ms)", "D2H-Pin(ms)", "D2H Speedup");
    printf("%-8s %14s %14s %10s %14s %14s %10s\n",
           "----",
           "----------", "----------", "-----------",
           "----------", "----------", "-----------");
    for (int i = 0; i < num_sizes; i++) {
        int N = pg_results[i].N;
        float h2d_speedup = pg_results[i].h2d_ms
                          / pin_results[i].h2d_ms;
        float d2h_speedup = pg_results[i].d2h_ms
                          / pin_results[i].d2h_ms;
        printf("%-8d %14.4f %14.4f %10.1fx %14.4f "
               "%14.4f %10.1fx\n",
               N,
               pg_results[i].h2d_ms,
               pin_results[i].h2d_ms,
               h2d_speedup,
               pg_results[i].d2h_ms,
               pin_results[i].d2h_ms,
               d2h_speedup);
    }

    // ---- Table 3: Bandwidth ----
    printf("\n[3] MEMORY BANDWIDTH (kernel only, T4 peak = %.0f GB/s)\n",
           peak_bw);
    printf("%-8s %14s %14s %10s\n",
           "N", "Pageable(GB/s)", "Pinned(GB/s)", "% of Peak");
    printf("%-8s %14s %14s %10s\n",
           "----", "-------------", "------------", "---------");
    for (int i = 0; i < num_sizes; i++) {
        int N = pg_results[i].N;
        float bw = pin_results[i].bandwidth_GBs; // pinned = cleaner
        printf("%-8d %14.1f %14.1f %9.1f%%\n",
               N,
               pg_results[i].bandwidth_GBs,
               pin_results[i].bandwidth_GBs,
               bw / peak_bw * 100.0f);
    }

    // ---- Table 4: Total Time (kernel + transfers) ----
    printf("\n[4] TOTAL TIME (kernel + H2D + D2H)\n");
    printf("%-8s %10s %14s %14s %12s %12s\n",
           "N", "CPU(ms)",
           "Total-Pg(ms)", "Total-Pin(ms)",
           "Speedup-Pg", "Speedup-Pin");
    printf("%-8s %10s %14s %14s %12s %12s\n",
           "----", "-------",
           "------------", "-------------",
           "----------", "-----------");
    for (int i = 0; i < num_sizes; i++) {
        int N = pg_results[i].N;
        printf("%-8d %10.4f %14.4f %14.4f %11.1fx %11.1fx\n",
               N,
               cpu_results[i],
               pg_results[i].total_ms,
               pin_results[i].total_ms,
               cpu_results[i] / pg_results[i].total_ms,
               cpu_results[i] / pin_results[i].total_ms);
    }

    // ---- Crossover Point ----
    printf("\n[5] CROSSOVER ANALYSIS\n");

    // Kernel crossover (where GPU kernel > CPU)
    int kernel_crossover = -1;
    for (int i = 0; i < num_sizes; i++) {
        if (pin_results[i].kernel_ms < cpu_results[i]) {
            kernel_crossover = pg_results[i].N;
            break;
        }
    }

    // Total crossover (where GPU total > CPU)
    int total_crossover = -1;
    for (int i = 0; i < num_sizes; i++) {
        if (pin_results[i].total_ms < cpu_results[i]) {
            total_crossover = pin_results[i].N;
            break;
        }
    }

    if (kernel_crossover > 0)
        printf("  Kernel crossover (GPU kernel < CPU):  N >= %d\n",
               kernel_crossover);
    else
        printf("  Kernel crossover: not reached in tested range\n");

    if (total_crossover > 0)
        printf("  Total crossover  (GPU total < CPU):   N >= %d\n",
               total_crossover);
    else
        printf("  Total crossover:  not reached in tested range\n");

    printf("\n  Key insight: large gap between kernel and total\n");
    printf("  crossover shows transfer overhead is significant!\n");
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[])
{
    // --------------------------------------------------------
    // WHICH N VALUES TO RUN?
    // --------------------------------------------------------
    // Default: run ALL standard sizes in one shot
    // Override: pass a single bit count to run just one size
    //
    // Usage:
    //   ./vector_add        → runs N = 10,16,20,22,24
    //   ./vector_add 22     → runs N = 1<<22 only
    // --------------------------------------------------------
    int default_bits[] = {10, 16, 20, 22, 24};
    int num_default    = 5;

    int*  bits_to_run;
    int   num_runs;
    int   single_bit;

    if (argc > 1) {
        // Single specific N
        single_bit  = atoi(argv[1]);
        bits_to_run = &single_bit;
        num_runs    = 1;
        if (single_bit < 1 || single_bit > 28) {
            fprintf(stderr,
                "Usage:\n"
                "  %s           (runs all sizes: 10,16,20,22,24)\n"
                "  %s <n_bits>  (runs single size: N = 1<<n_bits)\n",
                argv[0], argv[0]);
            return 1;
        }
    } else {
        // All default sizes
        bits_to_run = default_bits;
        num_runs    = num_default;
    }

    const int THREADS_PER_BLOCK = 256;

    // --------------------------------------------------------
    // PRINT HEADER ONCE
    // --------------------------------------------------------
    printf("============================================================\n");
    printf("HPS GPU Lab -- Experiment 1: Vector Addition\n");
    printf("Warmup runs: %d | Timed runs: %d | "
           "Threads/block: %d\n",
           WARMUP_RUNS, TIMED_RUNS, THREADS_PER_BLOCK);
    printf("============================================================\n");

    // GPU info (print once)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float peak_bw = 2.0f * prop.memoryClockRate
                  * (prop.memoryBusWidth / 8) / 1.0e6f;
    printf("GPU: %s | CC %d.%d | %d SMs | "
           "%.0f MB | Peak BW: %.0f GB/s\n",
           prop.name,
           prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6f,
           peak_bw);

    // --------------------------------------------------------
    // ALLOCATE RESULT ARRAYS
    // --------------------------------------------------------
    BenchResult* pg_results  = new BenchResult[num_runs];
    BenchResult* pin_results = new BenchResult[num_runs];
    float*       cpu_times   = new float[num_runs];

    // --------------------------------------------------------
    // MAIN LOOP OVER ALL N VALUES
    // --------------------------------------------------------
    for (int i = 0; i < num_runs; i++) {
        int n_bits = bits_to_run[i];
        int N      = 1 << n_bits;
        size_t bytes = (size_t)N * sizeof(float);

        printf("\n[N = %d  (1 << %d)  |  %.2f MB/array]\n",
               N, n_bits, bytes / 1e6f);

        // Allocate host memory for this N
        float* h_A     = (float*)malloc(bytes);
        float* h_B     = (float*)malloc(bytes);
        float* h_C_cpu = (float*)malloc(bytes);
        float* h_C_gpu = (float*)malloc(bytes);

        float* h_A_pin;
        float* h_B_pin;
        float* h_C_pin;
        CUDA_CHECK(cudaMallocHost(&h_A_pin, bytes));
        CUDA_CHECK(cudaMallocHost(&h_B_pin, bytes));
        CUDA_CHECK(cudaMallocHost(&h_C_pin, bytes));

        // Fill: A[i]=i, B[i]=2i → expected C[i]=3i
        for (int j = 0; j < N; j++) {
            h_A[j] = (float)j;
            h_B[j] = (float)(j * 2);
        }
        memcpy(h_A_pin, h_A, bytes);
        memcpy(h_B_pin, h_B, bytes);

        // CPU benchmark
        printf("  CPU...      ");
        fflush(stdout);
        cpu_times[i] = benchmark_cpu(h_A, h_B, h_C_cpu, N);
        printf("%.4f ms\n", cpu_times[i]);

        // GPU pageable benchmark
        printf("  Pageable... ");
        fflush(stdout);
        pg_results[i] = benchmark_gpu(h_A, h_B, h_C_gpu,
                                      N, THREADS_PER_BLOCK,
                                      cpu_times[i]);
        bool ok_pg = verify(h_C_cpu, h_C_gpu, N);
        printf("kernel=%.4f ms  H2D=%.4f ms  D2H=%.4f ms  "
               "BW=%.1f GB/s  [%s]\n",
               pg_results[i].kernel_ms,
               pg_results[i].h2d_ms,
               pg_results[i].d2h_ms,
               pg_results[i].bandwidth_GBs,
               ok_pg ? "OK" : "FAIL");

        // GPU pinned benchmark
        printf("  Pinned...   ");
        fflush(stdout);
        pin_results[i] = benchmark_gpu(h_A_pin, h_B_pin,
                                       h_C_pin, N,
                                       THREADS_PER_BLOCK,
                                       cpu_times[i]);
        bool ok_pin = verify(h_C_cpu, h_C_pin, N);
        printf("kernel=%.4f ms  H2D=%.4f ms  D2H=%.4f ms  "
               "BW=%.1f GB/s  [%s]\n",
               pin_results[i].kernel_ms,
               pin_results[i].h2d_ms,
               pin_results[i].d2h_ms,
               pin_results[i].bandwidth_GBs,
               ok_pin ? "OK" : "FAIL");

        // Free this N's host memory before next iteration
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        CUDA_CHECK(cudaFreeHost(h_A_pin));
        CUDA_CHECK(cudaFreeHost(h_B_pin));
        CUDA_CHECK(cudaFreeHost(h_C_pin));
    }

    // --------------------------------------------------------
    // PRINT FULL SUMMARY TABLE (all N values together)
    // --------------------------------------------------------
    print_summary_table(pg_results, pin_results,
                        cpu_times, bits_to_run,
                        num_runs, peak_bw);

    // Cleanup
    delete[] pg_results;
    delete[] pin_results;
    delete[] cpu_times;

    return 0;
}
