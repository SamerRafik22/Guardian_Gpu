#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include "cupti_monitor.h"
#include "stats.h"
#include "ipc_shared.h"

// Global Monitor Instance
CuptiMonitor g_cuptiMon;
HANDLE g_hPipe = INVALID_HANDLE_VALUE;

// Helper for DebugView logging AND Console
void ProbeLog(const std::string& msg) {
    std::string fullMsg = "[GuardianProbe] " + msg + "\n";
    OutputDebugStringA(fullMsg.c_str());
    // Also print to Host Console (if attached)
    std::cout << fullMsg; 
}

// Check CUDA Driver
#include <cuda.h> 
// Link against cuda.lib (Driver API) - tricky with CMake, let's use dynamic load or just rely on pure Runtime?
// Actually, cudaGetDeviceCount calls runtime which calls driver.
// Let's just use cudaFree(0) to force init context.

void ConnectToPipe() {
    ProbeLog("Attempting to connect to pipe: " + std::string(PIPE_NAME));
    // Try to connect to the pipe created by the injector
    for (int i = 0; i < 10; i++) { // Increased retries
        g_hPipe = CreateFileA(
            PIPE_NAME,
            GENERIC_WRITE,
            0,
            NULL,
            OPEN_EXISTING,
            0,
            NULL
        );
        
        if (g_hPipe != INVALID_HANDLE_VALUE) {
            ProbeLog("Pipe connected successfully!");
            break;
        }
        
        ProbeLog("Pipe connection attempt " + std::to_string(i+1) + " failed. Error: " + std::to_string(GetLastError()));
        Sleep(500);
    }
}

DWORD WINAPI MonitorThread(LPVOID lpParam) {
    ProbeLog("MonitorThread STARTED.");

    // 0. Force CUDA Context / Driver Init
    cudaFree(0); 
    ProbeLog("Forced CUDA Context Init (cudaFree(0)).");

    // 1. Initialize CUPTI
    ProbeLog("Calling g_cuptiMon.Initialize()...");
    if (!g_cuptiMon.Initialize()) {
        ProbeLog("FATAL: Failed to init CUPTI. Check CUPTI/CUDA DLLs.");
        return 1;
    }
    ProbeLog("CUPTI Initialized successfully.");

    ConnectToPipe();

    if (g_hPipe == INVALID_HANDLE_VALUE) {
        ProbeLog("FATAL: Failed to connect to pipe after retries.");
        return 1;
    }

    ProbeLog("Monitoring Loop Starting...");

    while (true) {
        Sleep(1000);

        // DEBUG: Self-Stimulation to verify Hooks
        // This should trigger a Runtime API Callback
        int devCount = 0;
        cudaError_t err = cudaGetDeviceCount(&devCount); 
        if (err == cudaSuccess) {
             // We don't log success here to avoid spamming user console, 
             // but we EXPECT to see a "Callback FIRED" in DebugView because of this call.
        } else {
             ProbeLog("Self-Test Failed: cudaGetDeviceCount returned " + std::to_string(err));
        }

        // 2. Collect Data
        std::vector<KernelRecord> records = g_cuptiMon.PopRecords();
        if (!records.empty()) {
            ProbeLog("Collected " + std::to_string(records.size()) + " records.");
        }
        
        // 3. Aggregate Stats
        AggregatedStats stats = StatsEngine::CalculateStats(records);

        // 4. Send to Host
        KernelStatsPackage pkg;
        pkg.processId = GetCurrentProcessId();
        pkg.kernelCount = stats.totalKernels; // Keep as is
        // HACK: Use durationMeanNs decimal part or something? 
        // No, let's update IPC struct. Wait, that requires recompiling Monitor too.
        // Let's repurpose 'durationStdDevNs' (double) to hold (double)RawEvents for a moment?
        // Risky.
        
        // Better: We send PID. Monitor sees PID.
        // We print "ProbeLog" to Console.
        
        // Let's just log it for now to prove it works.
        uint32_t totalEvents = CuptiMonitor::GetCallbackCount();
        if (totalEvents > 0) {
             // This will show up in CudaWorkload console!
             std::cout << "[GuardianProbe] RAW EVENTS DETECTED: " << totalEvents << std::endl;
        }
        pkg.durationMeanNs = stats.durationMeanNs;
        pkg.durationStdDevNs = stats.durationStdDevNs;
        pkg.avgOccupancy = stats.avgTheoreticalOccupancy;
        pkg.memcpyCount = CuptiMonitor::GetMemcpyCount(); // Fetch and reset
        
        // Send PID even if stats are empty, to keep connection alive
        DWORD written;
        BOOL success = WriteFile(
            g_hPipe,
            &pkg,
            sizeof(KernelStatsPackage),
            &written,
            NULL
        );

        if (!success) {
            ProbeLog("Pipe write failed (Broken Pipe). Exiting thread.");
            break; 
        }
    }

    CloseHandle(g_hPipe);
    ProbeLog("MonitorThread EXIT.");
    return 0;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    {
        // Use OutputDebugString directly here to confirm DLL load
        OutputDebugStringA("[GuardianProbe] DLL_PROCESS_ATTACH - Loading...\n");
        DisableThreadLibraryCalls(hModule);
        
        HANDLE hThread = CreateThread(NULL, 0, MonitorThread, NULL, 0, NULL);
        if (hThread) {
            OutputDebugStringA("[GuardianProbe] MonitorThread created successfully.\n");
            CloseHandle(hThread);
        } else {
            OutputDebugStringA("[GuardianProbe] FAILED to create MonitorThread.\n");
        }
        break;
    }
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
