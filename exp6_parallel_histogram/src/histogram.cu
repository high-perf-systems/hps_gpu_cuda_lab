#include <torch/extension.h>

__global__ void naive_kernel(
    const int* __restrict__ data,
    int* __restrict__ histogram,
    const int num_bins, 
    const int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int num = data[idx];
        if (num >= 0 && num < num_bins)
        {
            atomicAdd(&histogram[num], 1);
        }
    }
}

torch::Tensor launch_naive(torch::Tensor input, int num_bins)
{
    int N = input.size(0);
    torch::Tensor output = torch::zeros({num_bins,}, input.options());
    dim3 blk(256);
    dim3 grd((N + 256 - 1) / 256);
    naive_kernel<<<grd, blk>>>(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        num_bins,
        N
    );
    return output;
}

__global__ void privatized_kernel(
    const int* __restrict__ data,
    int* __restrict__ histogram,
    const int num_bins,
    const int N
)
{
    extern __shared__ int histo_s[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    {
        histo_s[bin] = 0;
    }
    __syncthreads();
    if (idx < N)
    {
        int num = data[idx];
        if (num >= 0 && num < num_bins)
        {
            atomicAdd(&histo_s[num], 1);
        }
    }
    __syncthreads();
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    {
        int bin_val = histo_s[bin];
        if (bin_val > 0)
            atomicAdd(&histogram[bin], bin_val);
    }
}

torch::Tensor launch_privatized(torch::Tensor input, int num_bins)
{
    int N = input.size(0);
    torch::Tensor output = torch::zeros({num_bins,}, input.options());
    dim3 blk(256);
    dim3 grd((N + 256 - 1) / 256);
    privatized_kernel<<<grd, blk, num_bins * sizeof(int)>>>(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        num_bins,
        N
    );
    return output;
}

__global__ void coarsening_kernel(
    const int* __restrict__ data,
    int* __restrict__ histogram,
    const int num_bins,
    const int N
)
{
    extern __shared__ int histo_s[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    {
        histo_s[bin] = 0;
    }
    __syncthreads();
    for (int i=idx; i < N; i += gridDim.x * blockDim.x)
    {
        int num = data[i];
        if (num >= 0 && num < num_bins)
        {
            atomicAdd(&histo_s[num], 1);
        }
    }
    __syncthreads();
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
    {
        int bin_val = histo_s[bin];
        if (bin_val > 0)
            atomicAdd(&histogram[bin], bin_val);
    }
}

__global__ void aggregation_kernel(
    const int* __restrict__ data,
    int* __restrict__ histogram,
    const int num_bins,
    const int N)
    {
        extern __shared__ int histo_s[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
        {
            histo_s[bin] = 0;
        }
        __syncthreads();
        int accumulator = 0;
        int prevIdx = -1;
        for (int i=idx;i<N;i+=blockDim.x * gridDim.x)
        {
            int num = data[i];
            if (num >= 0 && num < num_bins)
            {
                if (num == prevIdx) ++accumulator;
                else
                {
                    if (accumulator > 0)
                        atomicAdd(&histo_s[prevIdx], accumulator);
                    accumulator = 1;
                    prevIdx = num;
                }
            }
        }
        if (accumulator > 0)
            atomicAdd(&histo_s[prevIdx], accumulator);
        
        __syncthreads();

        for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
        {
            int val = histo_s[bin];
            if (val > 0) atomicAdd(&histogram[bin], val);
        }

    }



torch::Tensor launch_coarsening(torch::Tensor input, int num_bins)
{
    int N = input.size(0);
    torch::Tensor output = torch::zeros({num_bins,}, input.options());
    dim3 blk(256);
    int C_FACTOR = 8; // thread coarsening factor
    int elems_per_block = 256 * C_FACTOR;
    dim3 grd((N + elems_per_block - 1) / elems_per_block);
    coarsening_kernel<<<grd, blk, num_bins * sizeof(int)>>>(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        num_bins,
        N
    );
    return output;
}

torch::Tensor launch_aggregation(torch::Tensor input, int num_bins)
{
    int N = input.size(0);
    torch::Tensor output = torch::zeros({num_bins,}, input.options());
    dim3 blk(256);
    int C_FACTOR = 8; // thread coarsening factor
    int elems_per_block = 256 * C_FACTOR;
    dim3 grd((N + elems_per_block - 1) / elems_per_block);
    aggregation_kernel<<<grd, blk, num_bins * sizeof(int)>>>(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        num_bins,
        N
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("launch_naive", &launch_naive, "naive histogram kernel");
    m.def("launch_privatized",&launch_privatized, "private histogram kernel");
    m.def("launch_coarsening", &launch_coarsening, "thread coarsening histogram kernel");
    m.def("launch_aggregation", &launch_aggregation, "aggregation kernel");
}