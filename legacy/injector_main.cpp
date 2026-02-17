#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <nvml.h>
#include <tlhelp32.h>

#include "ipc_shared.h"
#include "logger.h"      // For GpuProcessSnapshot (NVML logic)
#include "csv_logger.h"  // For Logging

// Global Stats Map (Populated by Pipe, Read by UI)
KernelStatsPackage g_lastStats = {};
bool g_pipeConnected = false;

// -----------------------------------------------------------------------------
// INJECTION LOGIC
// -----------------------------------------------------------------------------
bool InjectDLL(DWORD pid, const std::string& dllPath) {
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    if (!hProcess) {
        std::cerr << "[Error] Failed to open process " << pid << std::endl;
        return false;
    }

    LPVOID pRemotePath = VirtualAllocEx(hProcess, NULL, dllPath.size() + 1, MEM_COMMIT, PAGE_READWRITE);
    WriteProcessMemory(hProcess, pRemotePath, dllPath.c_str(), dllPath.size() + 1, NULL);

    HANDLE hThread = CreateRemoteThread(hProcess, NULL, 0, 
        (LPTHREAD_START_ROUTINE)GetProcAddress(GetModuleHandleA("kernel32.dll"), "LoadLibraryA"), 
        pRemotePath, 0, NULL);

    if (hThread) {
        WaitForSingleObject(hThread, INFINITE);
        CloseHandle(hThread);
        std::cout << "[Success] Injected " << dllPath << " into PID " << pid << std::endl;
    } else {
        std::cerr << "[Error] Failed to CreateRemoteThread" << std::endl;
    }

    VirtualFreeEx(hProcess, pRemotePath, 0, MEM_RELEASE);
    CloseHandle(hProcess);
    return true;
}

DWORD GetPidByName(const std::string& procName) {
    PROCESSENTRY32 entry;
    entry.dwSize = sizeof(PROCESSENTRY32);
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, NULL);

    if (Process32First(snapshot, &entry)) {
        while (Process32Next(snapshot, &entry)) {
            if (std::string(entry.szExeFile) == procName) {
                CloseHandle(snapshot);
                return entry.th32ProcessID;
            }
        }
    }
    CloseHandle(snapshot);
    return 0;
}

// -----------------------------------------------------------------------------
// PIPE SERVER LOGIC
// -----------------------------------------------------------------------------
DWORD WINAPI PipeServerThread(LPVOID lpParam) {
    HANDLE hPipe = CreateNamedPipeA(
        PIPE_NAME,
        PIPE_ACCESS_INBOUND,
        PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
        1,
        PIPE_BUFFER_SIZE,
        PIPE_BUFFER_SIZE,
        0,
        NULL
    );

    if (hPipe == INVALID_HANDLE_VALUE) return 1;

    std::cout << "[Pipe] Waiting for client connection..." << std::endl;
    if (ConnectNamedPipe(hPipe, NULL) || GetLastError() == ERROR_PIPE_CONNECTED) {
        std::cout << "[Pipe] Client Connected!" << std::endl;
        g_pipeConnected = true;

        while (true) {
            KernelStatsPackage pkg;
            DWORD bytesRead;
            BOOL success = ReadFile(hPipe, &pkg, sizeof(pkg), &bytesRead, NULL);

            if (!success || bytesRead == 0) break; // Client disconnected

            // Update Global Stats
            g_lastStats = pkg;
        }
    }

    g_pipeConnected = false;
    CloseHandle(hPipe);
    return 0;
}

// -----------------------------------------------------------------------------
// MAIN LOOP
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: GuardianInjector.exe <TargetProcessName.exe>" << std::endl;
        // std::cout << "Example: GuardianInjector.exe javaw.exe" << std::endl;
        // return 1;
        // Dev Mode: default to "javaw.exe" or "CudaTestApp.exe"
    }

    std::string targetName = (argc >= 2) ? argv[1] : "javaw.exe"; // Default for testing
    
    // 1. Start Pipe Server
    CreateThread(NULL, 0, PipeServerThread, NULL, 0, NULL);

    // 2. Find & Inject Target
    std::cout << "[Info] Searching for process: " << targetName << "..." << std::endl;
    DWORD pid = 0;
    while (pid == 0) {
        pid = GetPidByName(targetName);
        if (pid == 0) Sleep(1000);
    }

    // Get Full Path to DLL (Assume in same dir)
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string exeDir = std::string(buffer);
    exeDir = exeDir.substr(0, exeDir.find_last_of("\\/"));
    std::string dllPath = exeDir + "\\GuardianProbe.dll";

    if (!InjectDLL(pid, dllPath)) {
        return 1;
    }

    // 3. Monitor Loop
    if (!GpuMonitor::Initialize()) {
        std::cerr << "[Error] NVML Init Failed" << std::endl;
    }

    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    CsvLogger logger;

    std::cout << "\n" << std::left << std::setw(10) << "TIME" << " | KERNELS/s | DUR(us)  | OCC%  | STATUS" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    while (true) {
        Sleep(1000);
        
        // Combine Data
        // NVML Data for specific PID
        GpuDeviceMetrics globalDev = GpuMonitor::CaptureDeviceMetrics(device);
        auto allProcs = GpuMonitor::CaptureProcessSnapshots(device);
        
        // Find our target process in NVML list
        GpuProcessSnapshot targetSnap;
        bool foundInNvml = false;
        for(auto& p : allProcs) {
            if(p.pid == pid) { targetSnap = p; foundInNvml = true; break; }
        }

        // Print Console Row
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::cout << std::put_time(&tm, "%H:%M:%S") << " | "
                  << std::setw(9) << g_lastStats.kernelCount << " | "
                  << std::setw(8) << std::fixed << std::setprecision(1) << (g_lastStats.durationMeanNs / 1000.0) << " | "
                  << std::setw(5) << (g_lastStats.avgOccupancy * 100.0) << " | "
                  << (g_pipeConnected ? "Connected" : "Waiting...")
                  << std::endl;

        // CSV Log (Only if connected or found in NVML)
        if (g_pipeConnected || foundInNvml) {
             logger.LogRow(
                "NOW", // Fix timestamp later
                pid, targetName,
                targetSnap.vramUsedBytes, globalDev.powerUsage, targetSnap.utilizationGpu,
                globalDev.temperature, globalDev.fanSpeed,
                globalDev.coreClock, globalDev.memClock,
                globalDev.pcieRx, globalDev.pcieTx,
                g_lastStats.kernelCount, g_lastStats.durationMeanNs, g_lastStats.durationStdDevNs,
                g_lastStats.avgOccupancy, g_lastStats.memcpyCount
             );
        }
    }

    return 0;
}
