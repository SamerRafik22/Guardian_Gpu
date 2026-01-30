#include <iostream>
#include <windows.h>

extern "C" void LaunchTestKernel();

int main() {
    std::cout << "[TestApp] Starting CUDA Stress Test (PID: " << GetCurrentProcessId() << ")..." << std::endl;
    std::cout << "[TestApp] Running kernels forever. Press Ctrl+C to stop." << std::endl;

    int counter = 0;
    while (true) {
        LaunchTestKernel();
        // Sleep(10); // Small sleep to not freeze system entirely, but kept tight to show high kernel rate
        // Actually, to show visible "Kernels/sec", we want a lot of them.
        // LaunchTestKernel does 1 kernel.
        // Let's do a batch
        for(int i=0; i<100; i++) LaunchTestKernel();
        
        counter++;
        if (counter % 10 == 0) std::cout << "." << std::flush;
        Sleep(10);
    }
    return 0;
}
