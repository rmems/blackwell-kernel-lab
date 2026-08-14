// Minimal L3 smoke for RTX 5080 (sm_120). Issue #21 / RM-488.
// Not a production kernel — proves this repo can build and run device code.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void bkl_hello_kernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("bkl device_hello sm_120 ok\n");
  }
}

int main() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  if (count < 1) {
    std::fprintf(stderr, "no CUDA devices\n");
    return 1;
  }

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  std::printf("device0: %s compute %d.%d\n", prop.name, prop.major, prop.minor);

  bkl_hello_kernel<<<1, 1>>>();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
