#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <iomanip>
#include <set>
#include <tlhelp32.h>
#include <fstream>
#include <sstream>
#include <map>
#include <mutex>
#include <algorithm>

#include <sddl.h> // For ConvertStringSecurityDescriptorToSecurityDescriptor
#include "gpu_mon.h"
#include "etw_monitor.h" 
#include "logger.h"
#include "csv_logger.h"

// -----------------------------------------------------------------------------
// GLOBAL STATE
// -----------------------------------------------------------------------------
// struct ProcessStats removed (Legacy Pipe Logic)

std::mutex g_statsMutex;

std::set<std::string> g_blacklist = {
    "csrss.exe", "dwm.exe", "smss.exe", "services.exe", 
    "lsass.exe", "wininit.exe", "svchost.exe", "RuntimeBroker.exe",
    "Registry", "System", "Idle", "Memory Compression",
    "GuardianMonitor.exe", "SleepyApp.exe" 
};

// -----------------------------------------------------------------------------
// MAIN MONITOR LOOP
// -----------------------------------------------------------------------------
int main() {
    std::cout << "[GuardianMonitor] Starting Phase 5: ETW Passive Monitoring..." << std::endl;

    // 1. Initialize NVML
    if (!GpuMonitor::Initialize()) {
        std::cerr << "[Error] NVML Init Failed!" << std::endl;
        return 1;
    }
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device); // Assume GPU 0

    // 2. Start ETW Session
    EtwMonitor g_etwMonitor;
    if (!g_etwMonitor.StartSession()) {
        std::cerr << "[Error] Failed to start ETW Session. Run as Admin?" << std::endl;
        // Proceed anyway? No, ETW is core now.
        return 1;
    }

    CsvLogger logger;

    std::cout << "\n" 
              << std::left << std::setw(10) << "TIME"
              << std::setw(8) << "PID"
              << std::setw(20) << "NAME"
              << std::setw(10) << "MEM(MB)"
              << std::setw(8) << "PWR(W)"
              << std::setw(12) << "GPU_TIME(ms)"
              << std::setw(8) << "COUNTS"
              << std::setw(8) << "NET_TX"  // New
              << std::setw(8) << "NET_RX"  // New
              << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

    while (true) {
        Sleep(1000); // 1 Second Interval

        // A. Scan Processes via NVML
        auto nvmlProcs = GpuMonitor::CaptureProcessSnapshots(device);
        auto deviceMetrics = GpuMonitor::CaptureDeviceMetrics(device);
        
        // B. Get ETW Stats (Passive)
        // Returns usage since last call
        auto etwStats = g_etwMonitor.PopProcessUsage(); 

        // C. Display & Log
        std::string timeStr = "NOW"; // Todo format
        
        bool printedHeader = false;

        for (const auto& p : nvmlProcs) {
            double gpuTimeMs = 0.0;
            uint32_t packetCount = 0;
            uint64_t netTx = 0;
            uint64_t netRx = 0;

            if (etwStats.count(p.pid)) {
                gpuTimeMs = etwStats[p.pid].busyTimeMs;
                packetCount = etwStats[p.pid].packetCount;
                netTx = etwStats[p.pid].netTxBytes;
                netRx = etwStats[p.pid].netRxBytes;
            }
            
            // Only print if active
            if (gpuTimeMs > 0.1 || p.vramUsedBytes > 100*1024*1024) {
                 std::cout << std::left << std::setw(10) << " "
                      << std::setw(8) << p.pid
                      << std::setw(20) << p.processName.substr(0, 19)
                      << std::setw(10) << (p.vramUsedBytes / 1024 / 1024)
                      << std::setw(8) << (deviceMetrics.powerUsage / 1000.0)
                      << std::setw(12) << std::fixed << std::setprecision(2) << gpuTimeMs
                      << std::setw(8) << packetCount
                      << std::setw(8) << netTx
                      << std::setw(8) << netRx
                      << std::endl;
            }

            // Log CSV (Simplified)
            logger.LogRow(
                timeStr,
                p.pid, p.processName,
                p.vramUsedBytes, (deviceMetrics.powerUsage / 1000.0), gpuTimeMs, packetCount,
                netTx, netRx
            );
        }
    }

    return 0;
}
