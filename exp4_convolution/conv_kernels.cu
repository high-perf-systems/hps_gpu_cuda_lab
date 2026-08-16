// convolution kernels matching the implementation of F.conv2d() with padding = 0
#include <torch/extension.h> // for interface with pytorch

#define MAXK 11

__constant__ float d_kernel[MAXK*MAXK];

// kernel 1. naive 2d convolution
__global__ void conv2d_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H,
    int W,
    int kH,
    int kW,
    int oH,
    int oW
)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= oH || col >= oW) return;
  float Pacc = 0.0f;
  for (int kh=0;kh < kH; ++kh)
  {
    for (int kw=0;kw < kW; ++kw)
    {
      Pacc += input[(row+kh) * W + col+kw] * kernel[kh*kW + kw];
    }
  }
  output[row * oW + col] = Pacc;
}

// kernel 2 : caching the kernels in constant cache
__global__ void conv2d_constant_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W,
    int kH,
    int kW,
    int oH,
    int oW
)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= oH || col >= oW) return;
  float Pacc = 0.0f;
  for (int kh=0;kh < kH; ++kh)
  {
    for (int kw=0;kw < kW; ++kw)
    {
      Pacc += input[(row+kh) * W + col+kw] * d_kernel[kh*kW + kw];
    }
  }
  output[row * oW + col] = Pacc;
}

// kernel 3 : tiled convolution
// we design the approach where the TILE_DIM is number of output cells written to per tile and the 
// number of input cells is TILE_DIM + K - 1
template <int TILE_DIM_Y, int TILE_DIM_X>
__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W,
    int kH,
    int kW,
    int oH,
    int oW
)
{
  constexpr int IN_TILE_X = TILE_DIM_X + MAXK - 1;
  constexpr int IN_TILE_Y = TILE_DIM_Y + MAXK - 1;
  __shared__ float s_tile[IN_TILE_Y][IN_TILE_X];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_DIM_Y + ty;
  int col = blockIdx.x * TILE_DIM_X + tx;

  for (int i=ty; i < IN_TILE_Y; i+=TILE_DIM_Y)
  {
    for (int j=tx; j < IN_TILE_X; j+=TILE_DIM_X)
    {
      int in_row = blockIdx.y * TILE_DIM_Y + i;
      int in_col = blockIdx.x * TILE_DIM_X + j;
      if (in_row < H && in_col < W)
      {
        s_tile[i][j] = input[in_row * W + in_col];
      }
      else
      {
        s_tile[i][j] = 0.0f;
      }
    }
  }
  __syncthreads();

  if (row < oH && col < oW)
  {
    float Pacc = 0.0f;
    for (int kh = 0; kh < kH; ++kh)
    {
      for (int kw = 0; kw < kW; ++kw)
      {
        Pacc += s_tile[ty+kh][tx+kw] * d_kernel[kh * kW + kw];
      }
    }
    output[row * oW + col] = Pacc;
  }  
}

//wrappers
// the python calls these wrappers to run the kernel
torch::Tensor launch_naive(
  torch::Tensor input, // 2D HXW float32 contiguous
  torch::Tensor kernel // 2D kHXkW float32 contiguous
)
{
  int H = input.size(0);
  int W = input.size(1);
  int kH = kernel.size(0);
  int kW = kernel.size(1);
  int oH = H - kH + 1;
  int oW = W - kW + 1;
  torch::Tensor output = torch::zeros({oH, oW}, input.options());

  int threadX = 16;
  int threadY = 16;
  int blockX = (oW + threadX - 1) / threadX;
  int blockY = (oH + threadY - 1) / threadY;

  dim3 block(threadX, threadY);
  dim3 grid(blockX, blockY);
  conv2d_naive_kernel<<<grid, block>>>(
    input.data_ptr<float>(),
    kernel.data_ptr<float>(),
    output.data_ptr<float>(),
    H,
    W,
    kH,
    kW,
    oH,
    oW
  );
  return output;
}
// function to copy tensor from device to constant memory of the device
void copyKernelToConstant(torch::Tensor kernel)
{
  cudaMemcpyToSymbol(
    d_kernel, // global constant memory
    kernel.data_ptr<float>(),
    kernel.numel() * sizeof(float),
    0,
    cudaMemcpyDeviceToDevice
  );
}
torch::Tensor launch_constantKernel(
  torch::Tensor input, // 2D HXW float32 contiguous
  int kH,
  int kW
)
{
  int H = input.size(0);
  int W = input.size(1);
  int oH = H - kH + 1;
  int oW = W - kW + 1;
  torch::Tensor output = torch::zeros({oH, oW}, input.options());

  int threadX = 16;
  int threadY = 16;
  int blockX = (oW + threadX - 1) / threadX;
  int blockY = (oH + threadY - 1) / threadY;

  dim3 block(threadX, threadY);
  dim3 grid(blockX, blockY);


  conv2d_constant_kernel<<<grid, block>>>(
    input.data_ptr<float>(),
    output.data_ptr<float>(),
    H,
    W,
    kH,
    kW,
    oH,
    oW
  );
  return output;
}

torch::Tensor launch_tiledKernel16(
  torch::Tensor input, // 2D HXW float32 contiguous
  int kH,
  int kW
)
{
  int H = input.size(0);
  int W = input.size(1);
  int oH = H - kH + 1;
  int oW = W - kW + 1;
  torch::Tensor output = torch::zeros({oH, oW}, input.options());

  int threadX = 16;
  int threadY = 16;
  int blockX = (oW + threadX - 1) / threadX;
  int blockY = (oH + threadY - 1) / threadY;

  dim3 block(threadX, threadY);
  dim3 grid(blockX, blockY);


  conv2d_tiled_kernel<16, 16><<<grid, block>>>(
    input.data_ptr<float>(),
    output.data_ptr<float>(),
    H,
    W,
    kH,
    kW,
    oH,
    oW
  );
  return output;
}

torch::Tensor launch_tiledKernel32(
  torch::Tensor input, // 2D HXW float32 contiguous
  int kH,
  int kW
)
{
  int H = input.size(0);
  int W = input.size(1);
  int oH = H - kH + 1;
  int oW = W - kW + 1;
  torch::Tensor output = torch::zeros({oH, oW}, input.options());

  int threadX = 32;
  int threadY = 32;
  int blockX = (oW + threadX - 1) / threadX;
  int blockY = (oH + threadY - 1) / threadY;

  dim3 block(threadX, threadY);
  dim3 grid(blockX, blockY);


  conv2d_tiled_kernel<32, 32><<<grid, block>>>(
    input.data_ptr<float>(),
    output.data_ptr<float>(),
    H,
    W,
    kH,
    kW,
    oH,
    oW
  );
  return output;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_naive", &launch_naive, "naive 2d convolution (global memory)");
  m.def("copyKernelToConstant", &copyKernelToConstant, "copy kernel tensor from global memory to cached constant memory");
  m.def("launch_constantKernel", &launch_constantKernel, "2d convolution with kernel in constant meory");
  m.def("launch_tiledKernel16", &launch_tiledKernel16, "2d convolution with tiled kernel of tile size 16");
  m.def("launch_tiledKernel32", &launch_tiledKernel32, "2d convolution with tiled kernel of tile size 32");
}
