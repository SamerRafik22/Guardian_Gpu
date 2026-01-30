#include <cuda_runtime.h>
#include <stdio.h>
#include <windows.h>
#include <math.h>

// Simple kernel to burn some GPU cycles
__global__ void BusyWorkKernel(float* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < iterations; ++i) {
            val = sinf(val) * cosf(val) + sqrtf(fabs(val));
        }
        data[idx] = val;
    }
}

int main() {
    printf("[CudaWorkload] Starting GPU Stress Test...\n");
    printf("[CudaWorkload] PID: %lu\n", GetCurrentProcessId());

    int size = 1024 * 1024; // 1M elements
    int bytes = size * sizeof(float);
    float* d_data;
    
    // Allocate memory
    cudaError_t err = cudaMalloc(&d_data, bytes);
    if (err != cudaSuccess) {
        printf("CUDA Malloc failed: %d\n", err);
        return 1;
    }
    
    // Run loop
    printf("[CudaWorkload] Launching kernels... (Press Ctrl+C to stop)\n");
    while (1) {
        // Launch a kernel
        BusyWorkKernel<<<256, 256>>>(d_data, size, 1000);
        
        // Sync to make sure it finishes (and to give the monitor time to catch it)
        cudaDeviceSynchronize();
        
        // Sleep a tiny bit to not freeze the whole system completely
        Sleep(50);
    }

    cudaFree(d_data);
    return 0;
}
