#pragma once
// 1. MUST BE AT THE VERY TOP to stop the "struct redefinition" errors
//#ifndef CUPTI_NO_REDEFINITIONS
//#define CUPTI_NO_REDEFINITIONS
//#endif

// 2. System/CUDA Headers
#include <cuda_runtime.h>
#include <cupti.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <atomic>

// 3. Data Structure
// Defined in stats.h now, but we need to include it
#include "stats.h"

// 4. Class Definition
class CuptiMonitor {
public:
    CuptiMonitor();
    ~CuptiMonitor();

    bool Initialize();
    void Shutdown();

    // Returns all accumulated records since last call and CLEARS internal buffer
    std::vector<KernelRecord> PopRecords();

private:
    // Internal storage
    static std::vector<KernelRecord> s_accumulatedRecords;
    static std::atomic<uint32_t> s_memcpyCount;

public:
    static void AddMemcpy() { s_memcpyCount++; }
    static uint32_t GetMemcpyCount() { return s_memcpyCount.exchange(0); }
    static uint32_t GetCallbackCount();

