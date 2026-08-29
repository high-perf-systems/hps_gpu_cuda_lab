// ============================================================
// Experiment 5: 3D Stencil (Finite Difference)
// hps_gpu_cuda_lab — github.com/high-perf-systems
//
// Versions implemented in this file (see problem.md "Versions to
// Implement" and notes.md "Kernel Versions" for the full story):
//   1. GPU Naive              — global memory only
//   2. GPU Constant Memory    — coefficients broadcast via __constant__
//   3. GPU Shared-Memory Tile — full 3D tile + halo, dynamic shared mem
//   4. GPU Thread Coarsening  — rolling window of fully-haloed x-y
//                                planes, swept through z (underperforms;
//                                see notes.md Finding 2 for why — kept
//                                here as the documented first attempt)
//   5. GPU Register Tiling    — one haloed x-y plane in shared memory +
//                                per-thread register z-window, templated
//                                on radius (the version that actually wins)
//
// Correctness reference: stencil_reference.py (torch.conv3d with a
// dense zero-padded kernel — see problem.md Concept 6 for why this
// is a correctness oracle only, not a performance target).
//
// Compiled via torch.utils.cpp_extension.load from a Colab notebook
// (stencil.ipynb), same workflow as exp4_convolution/conv_kernels.cu.
// ============================================================

#include <torch/extension.h>

#define MAXR 3   // max radius supported (6th-order stencil, matches problem.md)

__constant__ float d_coeffs[MAXR + 1];

// ============================================================
// Flat index helpers.
// idx3d:     row-major [D][H][W] global-memory layout.
// smem_idx:  row-major flat index into a 3D shared-memory tile.
// plane_idx: row-major flat index into a single 2D shared-memory plane.
// ============================================================
__device__ __forceinline__ size_t idx3d(int d, int r, int c, int H, int W)
{
    // Explicitly cast every dimension to size_t to prevent intermediate
    // integer overflow when H or W are large.
    return static_cast<size_t>(d) * static_cast<size_t>(H) * static_cast<size_t>(W) +
           static_cast<size_t>(r) * static_cast<size_t>(W) +
           static_cast<size_t>(c);
}

__device__ __forceinline__ int smem_idx(int i, int j, int k, int in_tile)
{
    return (i * in_tile + j) * in_tile + k;
}

__device__ __forceinline__ int plane_idx(int r, int c, int in_tile)
{
    return r * in_tile + c;
}

// ============================================================
// VERSION 1: GPU NAIVE
// One thread per output point. Reads its 6*radius+1 neighbors
// directly from global memory; coefficients passed as a small
// device array (length radius+1: [0]=center, [1..radius]=taps).
// ============================================================
__global__ void stencil_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ coeffs,   // [0]=center, [1..radius]=per-axis neighbor coeff
    float* __restrict__ output,
    int H, int W,     // input spatial dims (D not needed inside the kernel itself)
    int radius,
    int oD, int oH, int oW)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if (col >= oW || row >= oH || depth >= oD) return;

    // Valid-mode convention (matches exp4): output point (depth,row,col)
    // is centered on input point (depth+radius, row+radius, col+radius).
    int in_d = depth + radius;
    int in_r = row + radius;
    int in_c = col + radius;

    float acc = 0.0f;

    acc += input[idx3d(in_d, in_r, in_c, H, W)] * coeffs[0];
    for (int k = 1; k <= radius; ++k)
    {
        acc += coeffs[k] * input[idx3d(in_d + k, in_r, in_c, H, W)];
        acc += coeffs[k] * input[idx3d(in_d - k, in_r, in_c, H, W)];
        acc += coeffs[k] * input[idx3d(in_d, in_r + k, in_c, H, W)];
        acc += coeffs[k] * input[idx3d(in_d, in_r - k, in_c, H, W)];
        acc += coeffs[k] * input[idx3d(in_d, in_r, in_c + k, H, W)];
        acc += coeffs[k] * input[idx3d(in_d, in_r, in_c - k, H, W)];
    }

    output[idx3d(depth, row, col, oH, oW)] = acc;
}

torch::Tensor launch_naive(torch::Tensor input, torch::Tensor coeffs, int radius)
{
    int D = input.size(0);
    int H = input.size(1);
    int W = input.size(2);

    int oD = D - 2 * radius;
    int oH = H - 2 * radius;
    int oW = W - 2 * radius;

    // Return an empty tensor if output dimensions are non-positive
    if (oD <= 0 || oH <= 0 || oW <= 0) {
        return torch::zeros({0, 0, 0}, input.options());
    }

    torch::Tensor output = torch::zeros({oD, oH, oW}, input.options());

    dim3 blk(8, 8, 8);
    dim3 grd((oW + blk.x - 1) / blk.x, (oH + blk.y - 1) / blk.y, (oD + blk.z - 1) / blk.z);
    stencil_naive_kernel<<<grd, blk>>>(
        input.data_ptr<float>(),
        coeffs.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W, radius, oD, oH, oW);
    return output;
}

void copyCoeffsToConstant(torch::Tensor coeffs)
{
    cudaMemcpyToSymbol(
        d_coeffs,   // global constant memory
        coeffs.data_ptr<float>(),
        coeffs.numel() * sizeof(float),
        0,
        cudaMemcpyDeviceToDevice);
}

// ============================================================
// VERSION 2: GPU NAIVE + coeffs cached in constant memory
// Same structure as Version 1, but coefficients are read from
// d_coeffs (broadcast to a warp in one cycle) instead of being
// passed in as a per-thread global-memory array.
// ============================================================
__global__ void stencil_constmem_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W,
    int radius,
    int oD, int oH, int oW)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if (col >= oW || row >= oH || depth >= oD) return;

    int in_d = depth + radius;
    int in_r = row + radius;
    int in_c = col + radius;

    float acc = 0.0f;

    acc += input[idx3d(in_d, in_r, in_c, H, W)] * d_coeffs[0];
    for (int k = 1; k <= radius; ++k)
    {
        acc += d_coeffs[k] * input[idx3d(in_d + k, in_r, in_c, H, W)];
        acc += d_coeffs[k] * input[idx3d(in_d - k, in_r, in_c, H, W)];
        acc += d_coeffs[k] * input[idx3d(in_d, in_r + k, in_c, H, W)];
        acc += d_coeffs[k] * input[idx3d(in_d, in_r - k, in_c, H, W)];
        acc += d_coeffs[k] * input[idx3d(in_d, in_r, in_c + k, H, W)];
        acc += d_coeffs[k] * input[idx3d(in_d, in_r, in_c - k, H, W)];
    }

    output[idx3d(depth, row, col, oH, oW)] = acc;
}

torch::Tensor launch_constmem(torch::Tensor input, int radius)
{
    int D = input.size(0);
    int H = input.size(1);
    int W = input.size(2);

    int oD = D - 2 * radius;
    int oH = H - 2 * radius;
    int oW = W - 2 * radius;

    if (oD <= 0 || oH <= 0 || oW <= 0) {
        return torch::zeros({0, 0, 0}, input.options());
    }

    torch::Tensor output = torch::zeros({oD, oH, oW}, input.options());

    dim3 blk(8, 8, 8);
    dim3 grd((oW + blk.x - 1) / blk.x, (oH + blk.y - 1) / blk.y, (oD + blk.z - 1) / blk.z);

    stencil_constmem_kernel<<<grd, blk>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W, radius, oD, oH, oW);
    return output;
}

#define OUT_TILE 8

// ============================================================
// VERSION 3: Shared memory tiling + Version 2
// Block collaboratively loads a (OUT_TILE+2*radius)^3 haloed input
// cube into shared memory (dynamic shared mem, sized to the actual
// runtime radius — no over-fetch), then every thread computes from
// shared memory instead of global memory.
// Prediction (problem.md Q2/Q3): large speedup over naive, but risks
// hitting an occupancy wall at large tile/radius due to cubic halo
// growth in 3D. Actual result (notes.md Finding 1): never beats
// naive at any tested radius — L2 already absorbs most of naive's
// "redundant" reads at this working-set size.
// ============================================================
__global__ void stencil_smem_tiling(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W, int D,
    int radius, int in_tile,
    int oD, int oH, int oW)
{
    extern __shared__ float s_mem[];
    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int col = blockIdx.x * OUT_TILE + tx;
    int row = blockIdx.y * OUT_TILE + ty;
    int depth = blockIdx.z * OUT_TILE + tz;

    int global_base_d = blockIdx.z * OUT_TILE;
    int global_base_r = blockIdx.y * OUT_TILE;
    int global_base_c = blockIdx.x * OUT_TILE;

    // Collaboratively fill the whole haloed cube — every thread loads
    // multiple cells (in_tile > OUT_TILE), stepping by OUT_TILE.
    for (int i = tz; i < in_tile; i += OUT_TILE)
    {
        for (int j = ty; j < in_tile; j += OUT_TILE)
        {
            for (int k = tx; k < in_tile; k += OUT_TILE)
            {
                int g_d = global_base_d + i;
                int g_r = global_base_r + j;
                int g_c = global_base_c + k;

                if (g_d < D && g_r < H && g_c < W)
                {
                    s_mem[smem_idx(i, j, k, in_tile)] = input[idx3d(g_d, g_r, g_c, H, W)];
                }
                else
                {
                    s_mem[smem_idx(i, j, k, in_tile)] = 0.0f;
                }
            }
        }
    }

    __syncthreads();

    // Out-of-bounds check for global output writing — placed AFTER the
    // sync above so every thread still participates in the collaborative
    // load, even ones that won't write an output point.
    if (col >= oW || row >= oH || depth >= oD) return;

    int s_d = tz + radius;
    int s_r = ty + radius;
    int s_c = tx + radius;
    float acc = s_mem[smem_idx(s_d, s_r, s_c, in_tile)] * d_coeffs[0];

    for (int k = 1; k <= radius; ++k)
    {
        acc += d_coeffs[k] * s_mem[smem_idx(s_d + k, s_r, s_c, in_tile)];
        acc += d_coeffs[k] * s_mem[smem_idx(s_d - k, s_r, s_c, in_tile)];
        acc += d_coeffs[k] * s_mem[smem_idx(s_d, s_r + k, s_c, in_tile)];
        acc += d_coeffs[k] * s_mem[smem_idx(s_d, s_r - k, s_c, in_tile)];
        acc += d_coeffs[k] * s_mem[smem_idx(s_d, s_r, s_c + k, in_tile)];
        acc += d_coeffs[k] * s_mem[smem_idx(s_d, s_r, s_c - k, in_tile)];
    }

    output[idx3d(depth, row, col, oH, oW)] = acc;
}

torch::Tensor launch_smemtile(torch::Tensor input, int radius)
{
    int D = input.size(0);
    int H = input.size(1);
    int W = input.size(2);

    int oD = D - 2 * radius;
    int oH = H - 2 * radius;
    int oW = W - 2 * radius;

    if (oD <= 0 || oH <= 0 || oW <= 0) {
        return torch::zeros({0, 0, 0}, input.options());
    }

    torch::Tensor output = torch::zeros({oD, oH, oW}, input.options());

    dim3 blk(8, 8, 8);
    dim3 grd((oW + blk.x - 1) / blk.x, (oH + blk.y - 1) / blk.y, (oD + blk.z - 1) / blk.z);

    // Sized to the ACTUAL runtime radius, not MAXR — avoids loading a
    // larger halo than the current call actually needs.
    int in_tile = OUT_TILE + 2 * radius;

    stencil_smem_tiling<<<grd, blk, in_tile * in_tile * in_tile * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W, D, radius, in_tile, oD, oH, oW);
    return output;
}

#define OUT_TILE_TC 16

// ============================================================
// VERSION 4: Thread Coarsening
// A 2D (x,y) block sweeps a chunk of z-depths in a loop instead of
// using a 3rd block dimension, keeping a rolling window of
// `2*radius+1` FULLY HALOED x-y planes in shared memory (drop the
// oldest plane, load one new leading plane, per z-step).
//
// Grid is chunked in z (gridDim.z = ceil(oD/OUT_TILE_TC)) — a single
// block sweeping the *entire* z-depth collapses the block count and
// starves the GPU of parallelism (see notes.md Finding 2 for the
// numbers). Chunking fixes that, but a second, larger problem
// remains: only the CENTER plane's halo is ever read (z-taps always
// read the same (row,col) as center, never an offset one, since the
// stencil is an axis-aligned cross) — so the other `2*radius` planes
// here are loading a halo margin that is provably never used. That
// waste grows with radius, which is why this version gets WORSE
// relative to naive as radius increases. See Version 5 for the fix.
// ============================================================
__global__ void stencil_thread_coarse(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W, int D,
    int radius, int in_tile,
    int oD, int oH, int oW)
{
    extern __shared__ float s_mem[];
    int plane_elems = in_tile * in_tile;
    int n_planes = 2 * radius + 1;

    float* planes[2 * MAXR + 1];   // allocating for max radius case
    for (int p = 0; p < n_planes; ++p) planes[p] = s_mem + p * plane_elems;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int col = blockIdx.x * OUT_TILE_TC + tx;
    int row = blockIdx.y * OUT_TILE_TC + ty;

    int global_base_r = blockIdx.y * OUT_TILE_TC;
    int global_base_c = blockIdx.x * OUT_TILE_TC;

    // initial fill : z = z_start to z_start + 2*radius
    int z_start = blockIdx.z * OUT_TILE_TC;
    for (int z = z_start; z < (z_start + 2 * radius + 1); ++z)
    {
        for (int i = ty; i < in_tile; i += OUT_TILE_TC)
        {
            for (int j = tx; j < in_tile; j += OUT_TILE_TC)
            {
                int g_r = global_base_r + i;
                int g_c = global_base_c + j;
                if (z < D && g_r < H && g_c < W)
                {
                    planes[z - z_start][plane_idx(i, j, in_tile)] = input[idx3d(z, g_r, g_c, H, W)];
                }
                else
                {
                    planes[z - z_start][plane_idx(i, j, in_tile)] = 0.0f;
                }
            }
        }
    }

    int s_r = ty + radius;
    int s_c = tx + radius;
    for (int zc = 0; zc < OUT_TILE_TC; ++zc)
    {
        __syncthreads();

        int z = z_start + zc;
        if (z >= oD) break;   // uniform across the block -> safe, no barrier divergence

        float acc = planes[radius][plane_idx(s_r, s_c, in_tile)] * d_coeffs[0];
        for (int k = 1; k <= radius; ++k)
        {
            acc += d_coeffs[k] * planes[radius][plane_idx(s_r, s_c + k, in_tile)];
            acc += d_coeffs[k] * planes[radius][plane_idx(s_r, s_c - k, in_tile)];
            acc += d_coeffs[k] * planes[radius][plane_idx(s_r + k, s_c, in_tile)];
            acc += d_coeffs[k] * planes[radius][plane_idx(s_r - k, s_c, in_tile)];
            acc += d_coeffs[k] * planes[radius - k][plane_idx(s_r, s_c, in_tile)];
            acc += d_coeffs[k] * planes[radius + k][plane_idx(s_r, s_c, in_tile)];
        }
        // guard the write only — every thread must still participate in
        // every __syncthreads() below, regardless of (col,row) bounds.
        if (col < oW && row < oH)
            output[idx3d(z, row, col, oH, oW)] = acc;
        __syncthreads();

        // slide the rolling window: drop the oldest plane, reuse its
        // storage for the new leading plane.
        float* dropped = planes[0];
        for (int p = 0; p < n_planes - 1; ++p) planes[p] = planes[p + 1];
        planes[n_planes - 1] = dropped;

        int newD = z + 2 * radius + 1;

        for (int i = ty; i < in_tile; i += OUT_TILE_TC)
        {
            for (int j = tx; j < in_tile; j += OUT_TILE_TC)
            {
                int g_r = global_base_r + i;
                int g_c = global_base_c + j;
                if (newD < D && g_r < H && g_c < W)
                {
                    planes[n_planes - 1][plane_idx(i, j, in_tile)] = input[idx3d(newD, g_r, g_c, H, W)];
                }
                else
                {
                    planes[n_planes - 1][plane_idx(i, j, in_tile)] = 0.0f;
                }
            }
        }
    }
}

torch::Tensor launch_threadcoarse(torch::Tensor input, int radius)
{
    int D = input.size(0);
    int H = input.size(1);
    int W = input.size(2);

    int oD = D - 2 * radius;
    int oH = H - 2 * radius;
    int oW = W - 2 * radius;

    if (oD <= 0 || oH <= 0 || oW <= 0) {
        return torch::zeros({0, 0, 0}, input.options());
    }

    torch::Tensor output = torch::zeros({oD, oH, oW}, input.options());

    dim3 blk(OUT_TILE_TC, OUT_TILE_TC);
    dim3 grd((oW + blk.x - 1) / blk.x, (oH + blk.y - 1) / blk.y, (oD + OUT_TILE_TC - 1) / OUT_TILE_TC);

    int in_tile = OUT_TILE_TC + 2 * radius;

    stencil_thread_coarse<<<grd, blk, (2 * radius + 1) * in_tile * in_tile * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W, D, radius, in_tile, oD, oH, oW);
    return output;
}

// ============================================================
// VERSION 5: Register Tiling
//
// Only the x/y in-plane neighbors need a halo (they read
// neighboring columns/rows). The z neighbors always read the
// exact same (row, col) as the center — no halo needed for
// them at all. So: keep ONE haloed x-y plane in shared memory
// (the current z-center), and keep the z-window as a small
// per-thread array that (thanks to templating on RADIUS below)
// the compiler can fully unroll and allocate in registers,
// instead of a shared/synchronized buffer.
//
// Templated on RADIUS so `zreg`'s size and the tap loops are
// compile-time constants — a runtime `radius` loop bound would
// force zreg into local memory (global-memory-backed), defeating
// the point of "register" tiling.
//
// Result (notes.md Finding 3): the only version that beats naive at
// every radius, with the margin GROWING with radius — the opposite
// trend of Version 4, because this design eliminates exactly the
// wasted halo loads that made V4 degrade with radius.
// ============================================================

#define Z_CHUNK OUT_TILE_TC   // z-depth each block sweeps; independent knob from the x/y tile size — tune separately

template <int RADIUS>
__global__ void stencil_register_tile(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W, int D,
    int in_tile,     // OUT_TILE_TC + 2*RADIUS, x/y halo tile width
    int oD, int oH, int oW)
{
    extern __shared__ float s_mem[];     // ONE haloed x-y plane — the current z-center only
    float zreg[2 * RADIUS + 1];          // per-thread z-window; compile-time sized -> registers

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int col = blockIdx.x * OUT_TILE_TC + tx;   // output-space column
    int row = blockIdx.y * OUT_TILE_TC + ty;   // output-space row

    // input-space position for THIS thread's own column — the z-taps always
    // read here, never offset, since the stencil is an axis-aligned cross.
    int in_row = row + RADIUS;
    int in_col = col + RADIUS;

    int global_base_r = blockIdx.y * OUT_TILE_TC;
    int global_base_c = blockIdx.x * OUT_TILE_TC;

    int z_start = blockIdx.z * Z_CHUNK;   // first output depth this block owns

    // ---- initial fill: z-window covering input depths [z_start, z_start+2*RADIUS] ----
    // zreg[RADIUS] ends up holding the center depth for output z = z_start.
    #pragma unroll
    for (int p = 0; p < 2 * RADIUS + 1; ++p) {
        int z = z_start + p;
        zreg[p] = (z < D && in_row < H && in_col < W)
                      ? input[idx3d(z, in_row, in_col, H, W)]
                      : 0.0f;
    }

    // ---- initial fill: haloed x-y tile for the center plane (z_start+RADIUS) ----
    for (int i = ty; i < in_tile; i += OUT_TILE_TC) {
        for (int j = tx; j < in_tile; j += OUT_TILE_TC) {
            int g_r = global_base_r + i;
            int g_c = global_base_c + j;
            s_mem[plane_idx(i, j, in_tile)] =
                (z_start + RADIUS < D && g_r < H && g_c < W)
                    ? input[idx3d(z_start + RADIUS, g_r, g_c, H, W)]
                    : 0.0f;
        }
    }

    int s_r = ty + RADIUS;   // this thread's row inside the shared tile
    int s_c = tx + RADIUS;   // this thread's col inside the shared tile

    // ---- sweep this block's z-chunk ----
    for (int zc = 0; zc < Z_CHUNK; ++zc) {
        __syncthreads();   // s_mem (and zreg, on the first pass) must be fully loaded first

        int z = z_start + zc;
        if (z >= oD) break;   // same for every thread in the block -> safe, no barrier divergence

        // center tap
        float acc = s_mem[plane_idx(s_r, s_c, in_tile)] * d_coeffs[0];

        #pragma unroll
        for (int k = 1; k <= RADIUS; ++k) {
            acc += d_coeffs[k] * s_mem[plane_idx(s_r, s_c + k, in_tile)];  // +x, shared
            acc += d_coeffs[k] * s_mem[plane_idx(s_r, s_c - k, in_tile)];  // -x, shared
            acc += d_coeffs[k] * s_mem[plane_idx(s_r + k, s_c, in_tile)];  // +y, shared
            acc += d_coeffs[k] * s_mem[plane_idx(s_r - k, s_c, in_tile)];  // -y, shared
            acc += d_coeffs[k] * zreg[RADIUS - k];                        // -z, register
            acc += d_coeffs[k] * zreg[RADIUS + k];                        // +z, register
        }

        if (col < oW && row < oH)
            output[idx3d(z, row, col, oH, oW)] = acc;

        __syncthreads();   // everyone must finish reading s_mem before it gets overwritten below

        // ---- slide the z-window by one: drop oldest, append the new leading edge ----
        #pragma unroll
        for (int p = 0; p < 2 * RADIUS; ++p) zreg[p] = zreg[p + 1];

        int new_z_tap = z + 2 * RADIUS + 1;   // one past the window's previous far edge
        zreg[2 * RADIUS] = (new_z_tap < D && in_row < H && in_col < W)
                                ? input[idx3d(new_z_tap, in_row, in_col, H, W)]
                                : 0.0f;

        // ---- slide the shared tile forward by exactly one plane ----
        int new_center = z + RADIUS + 1;
        for (int i = ty; i < in_tile; i += OUT_TILE_TC) {
            for (int j = tx; j < in_tile; j += OUT_TILE_TC) {
                int g_r = global_base_r + i;
                int g_c = global_base_c + j;
                s_mem[plane_idx(i, j, in_tile)] =
                    (new_center < D && g_r < H && g_c < W)
                        ? input[idx3d(new_center, g_r, g_c, H, W)]
                        : 0.0f;
            }
        }
    }
}

// ============================================================
// WRAPPER — dispatches to the matching template instantiation
// since `radius` only exists at runtime here but the kernel
// needs it at compile time.
// ============================================================
torch::Tensor launch_registertile(torch::Tensor input, int radius)
{
    int D = input.size(0);
    int H = input.size(1);
    int W = input.size(2);

    int oD = D - 2 * radius;
    int oH = H - 2 * radius;
    int oW = W - 2 * radius;

    if (oD <= 0 || oH <= 0 || oW <= 0)
        return torch::zeros({0, 0, 0}, input.options());

    torch::Tensor output = torch::zeros({oD, oH, oW}, input.options());

    int in_tile = OUT_TILE_TC + 2 * radius;
    size_t smem_bytes = (size_t)in_tile * in_tile * sizeof(float);   // ONE plane, not (2r+1)

    dim3 blk(OUT_TILE_TC, OUT_TILE_TC);
    dim3 grd(
        (oW + OUT_TILE_TC - 1) / OUT_TILE_TC,
        (oH + OUT_TILE_TC - 1) / OUT_TILE_TC,
        (oD + Z_CHUNK - 1) / Z_CHUNK);

    switch (radius) {
        case 1:
            stencil_register_tile<1><<<grd, blk, smem_bytes>>>(
                input.data_ptr<float>(), output.data_ptr<float>(), H, W, D, in_tile, oD, oH, oW);
            break;
        case 2:
            stencil_register_tile<2><<<grd, blk, smem_bytes>>>(
                input.data_ptr<float>(), output.data_ptr<float>(), H, W, D, in_tile, oD, oH, oW);
            break;
        case 3:
            stencil_register_tile<3><<<grd, blk, smem_bytes>>>(
                input.data_ptr<float>(), output.data_ptr<float>(), H, W, D, in_tile, oD, oH, oW);
            break;
        default:
            TORCH_CHECK(false, "launch_registertile: radius must be 1..", MAXR);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_naive", &launch_naive, "naive 3D stencil (global memory)");
    m.def("copyCoeffsToConstant", &copyCoeffsToConstant, "copy coeffs to constant memory");
    m.def("launch_constmem", &launch_constmem, "naive 3D stencil (global memory), constant coeffs");
    m.def("launch_smemtile", &launch_smemtile, "shared memory tiling");
    m.def("launch_threadcoarse", &launch_threadcoarse, "thread coarsening");
    m.def("launch_registertile", &launch_registertile, "register tiling");
}
