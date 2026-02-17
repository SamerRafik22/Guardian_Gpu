// 1. MUST BE FIRST
#define __generated_cuda_runtime_api_meta_h__ 

// 2. System and CUDA Headers
#include <cuda_runtime.h>
#include <cupti.h>
#include <vector>
#include <iostream>
#include <mutex>
#include <map>
#include <chrono>
#include <windows.h>
#include <atomic>
#include <sstream>

// 3. Your Headers
#include "cupti_monitor.h"

// -----------------------------------------------------------------------------
// GLOBAL STATE
// -----------------------------------------------------------------------------
std::vector<KernelRecord> CuptiMonitor::s_accumulatedRecords;
std::mutex g_recordsMutex;
CUpti_SubscriberHandle g_subscriber = nullptr;
std::atomic<uint32_t> CuptiMonitor::s_memcpyCount{0};

// Expose the global counter via class method
// We need to make g_callbackFireCount atomic or thread-safe if we want to read it cleanly from another thread
extern std::atomic<uint32_t> g_callbackFireCountAtomic; // Declare extern
std::atomic<uint32_t> g_callbackFireCountAtomic{0}; // Define

uint32_t CuptiMonitor::GetCallbackCount() {
    return g_callbackFireCountAtomic.load();
}


// Kernel tracking
struct PendingKernel {
    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    std::string name;
    uint64_t launchTime;
};

std::map<uint64_t, PendingKernel> g_pendingKernels;
uint64_t g_kernelIdCounter = 0;
std::mutex g_kernelsMutex;

// Diagnostic counters
// uint32_t g_callbackFireCount = 0; // Replaced by atomic above

void DebugLog(const std::string& msg) {
    OutputDebugStringA(("[CUPTI] " + msg).c_str());
    std::cout << "[CUPTI] " << msg << std::endl;
}

// -----------------------------------------------------------------------------
// CALLBACK HANDLER
// -----------------------------------------------------------------------------
void CUPTIAPI CallbackHandler(
    void* userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void* cbdata)
{
    g_callbackFireCountAtomic++;
    
    std::ostringstream oss;
    oss << "Callback FIRED! Count=" << g_callbackFireCountAtomic.load() 
        << " Domain=" << domain << " CBID=" << cbid;
    DebugLog(oss.str());

    const CUpti_CallbackData* cbInfo = (const CUpti_CallbackData*)cbdata;

    // Log ALL callbacks to see what's happening
    if (cbInfo->symbolName) {
        DebugLog(std::string("  Symbol: ") + cbInfo->symbolName);
    }

    // Handle kernel launches
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        // Check for ANY launch-related callback
        bool isLaunch = (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
                         cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);

        if (isLaunch) {
            if (cbInfo->callbackSite == CUPTI_API_ENTER) {
                DebugLog("  -> KERNEL LAUNCH ENTER");
                // ... (Existing Launch Logic) ...
                PendingKernel pk;
                cudaEventCreate(&pk.startEvent);
                cudaEventCreate(&pk.endEvent);
                cudaEventRecord(pk.startEvent, 0);

                if (cbInfo->symbolName) {
                    pk.name = std::string(cbInfo->symbolName);
                } else {
                    pk.name = "Unknown";
                }
                pk.launchTime = std::chrono::high_resolution_clock::now().time_since_epoch().count();

                std::lock_guard<std::mutex> lock(g_kernelsMutex);
                uint64_t id = g_kernelIdCounter++;
                g_pendingKernels[id] = pk;
            }
            else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                DebugLog("  -> KERNEL LAUNCH EXIT");
                
                std::lock_guard<std::mutex> lock(g_kernelsMutex);
                if (!g_pendingKernels.empty()) {
                    auto it = g_pendingKernels.rbegin();
                    cudaEventRecord(it->second.endEvent, 0);
                }
            }
        }
        else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                CuptiMonitor::AddMemcpy();
                DebugLog("  -> MEMCPY DETECTED (Runtime)!");
            }
        }
    }
    // Handle DRIVER API (for static apps)
    else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        // Any function call here proves activity
        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            // Just count it as a "Raw Event" for now to prove life
            // Ideally check for cuMemcpy...
             DebugLog(std::string("  -> DRIVER API CALL: ") + (cbInfo->symbolName ? cbInfo->symbolName : "Unknown"));
        }
    }
}

// -----------------------------------------------------------------------------
// CuptiMonitor Implementation
// -----------------------------------------------------------------------------
CuptiMonitor::CuptiMonitor() {}
CuptiMonitor::~CuptiMonitor() { Shutdown(); }

bool CuptiMonitor::Initialize() {
    DebugLog("=== CUPTI Initialize START ===");
    
    // Subscribe to callbacks
    CUptiResult res = cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)CallbackHandler, nullptr);
    if (res != CUPTI_SUCCESS) {
        DebugLog("FAILED cuptiSubscribe: " + std::to_string(res));
        return false;
    }
    DebugLog("SUCCESS cuptiSubscribe");

    // Try to enable ALL possible launch callbacks
    // DEBUG: Enable ENTIRE Runtime Domain to see what is happening
    res = cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    if (res != CUPTI_SUCCESS) {
         DebugLog("FAILED to enable Runtime Domain: " + std::to_string(res));
    } else {
         DebugLog("SUCCESS: Enabled ENTIRE Runtime Domain");
    }

    // ENABLE DRIVER DOMAIN (Crucial for Statically Linked Apps)
    res = cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    if (res != CUPTI_SUCCESS) {
         DebugLog("FAILED to enable Driver Domain: " + std::to_string(res));
    } else {
         DebugLog("SUCCESS: Enabled ENTIRE Driver Domain");
    }

    return true;
}

void CuptiMonitor::Shutdown() {
    DebugLog("=== CUPTI Shutdown ===");
    if (g_subscriber) {
        cuptiUnsubscribe(g_subscriber);
        g_subscriber = nullptr;
    }

    std::lock_guard<std::mutex> lock(g_kernelsMutex);
    for (auto& kv : g_pendingKernels) {
        cudaEventDestroy(kv.second.startEvent);
        cudaEventDestroy(kv.second.endEvent);
    }
    g_pendingKernels.clear();
}

std::vector<KernelRecord> CuptiMonitor::PopRecords() {
    std::vector<KernelRecord> output;

    {
        std::lock_guard<std::mutex> lock(g_kernelsMutex);
        for (auto it = g_pendingKernels.begin(); it != g_pendingKernels.end(); ) {
            auto& pk = it->second;

            cudaError_t status = cudaEventQuery(pk.endEvent);
            if (status == cudaSuccess) {
                float durationMs = 0;
                cudaEventElapsedTime(&durationMs, pk.startEvent, pk.endEvent);

                KernelRecord kr;
                kr.name = pk.name;
                kr.durationNs = durationMs * 1e6;
                kr.startTimestamp = pk.launchTime;
                kr.endTimestamp = pk.launchTime + (uint64_t)(durationMs * 1e6);
                kr.deviceId = 0;
                kr.processId = 0;
                kr.gridX = kr.gridY = kr.gridZ = 1;
                kr.blockX = kr.blockY = kr.blockZ = 1;
                kr.numRegisters = 0;
                kr.sharedMemoryBytes = 0;

                cudaEventDestroy(pk.startEvent);
                cudaEventDestroy(pk.endEvent);

                output.push_back(kr);
                it = g_pendingKernels.erase(it);
                
                DebugLog("Finalized kernel: " + kr.name);
            } else {
                ++it;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_recordsMutex);
        s_accumulatedRecords.insert(s_accumulatedRecords.end(), output.begin(), output.end());
    }

    std::vector<KernelRecord> ret;
    {
        std::lock_guard<std::mutex> lock(g_recordsMutex);
        ret = s_accumulatedRecords;
        s_accumulatedRecords.clear();
    }

    if (!ret.empty()) {
        DebugLog("PopRecords returning " + std::to_string(ret.size()) + " kernels");
    }

    return ret;
}