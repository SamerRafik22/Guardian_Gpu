#include <iostream>
#include <windows.h>
#include <cuda_runtime.h>

// Link against cudart via CMake

int main() {
    std::cout << "[CudaWorkload] Starting GPU Memory Stress Test (Pure C++)..." << std::endl;
    std::cout << "[CudaWorkload] PID: " << GetCurrentProcessId() << std::endl;

    size_t size = 1024 * 1024 * 10; // 10 MB
    void* devPtr = nullptr;
    void* hostPtr = malloc(size);
    memset(hostPtr, 1, size);

    std::cout << "[CudaWorkload] Looping cudaMalloc/Memcpy/Free..." << std::endl;

    while (true) {
        cudaError_t err = cudaMalloc(&devPtr, size);
        if (err != cudaSuccess) {
            std::cerr << "Malloc failed: " << err << std::endl;
            break;
        }

        err = cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
             std::cerr << "Memcpy H2D failed: " << err << std::endl;
             cudaFree(devPtr);
             break;
        }
        
        err = cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
         if (err != cudaSuccess) {
             std::cerr << "Memcpy D2H failed: " << err << std::endl;
             cudaFree(devPtr);
             break;
        }

        cudaFree(devPtr);
        Sleep(100); // 10ms
    }

    free(hostPtr);
    return 0;
}
