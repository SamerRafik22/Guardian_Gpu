#include <cuda_runtime.h>
#include <stdio.h>

__global__ void TestKernel(float* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for(int i=0; i<1000; i++) { 
             d_data[idx] = d_data[idx] * d_data[idx] + 0.01f;
        }
    }
}

extern "C" void LaunchTestKernel() {
    float* d_data = nullptr;
    cudaMalloc(&d_data, 1024*sizeof(float));
    TestKernel<<<1, 256>>>(d_data, 1024);
    cudaDeviceSynchronize();
    cudaFree(d_data);
}
