#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdint>

struct KernelRecord {
    std::string name;
    uint64_t startTimestamp;
    uint64_t endTimestamp;
    uint64_t durationNs;
    uint32_t deviceId;
    uint32_t processId;
    
    // For Occupancy
    uint32_t numRegisters;
    uint32_t sharedMemoryBytes;
    uint32_t blockX, blockY, blockZ;
    uint32_t gridX, gridY, gridZ;
};

struct AggregatedStats {
    uint32_t totalKernels = 0;
    double durationMeanNs = 0.0;
    double durationStdDevNs = 0.0;
    double avgTheoreticalOccupancy = 0.0;
    uint32_t memcpyCount = 0;
    double memcpyThroughputMB = 0.0; // Approximation if bytes known
};

class StatsEngine {
public:
    static AggregatedStats CalculateStats(const std::vector<KernelRecord>& records) {
        AggregatedStats stats = {};
        if (records.empty()) return stats;

        stats.totalKernels = (uint32_t)records.size();
        
        // 1. Calculate Mean Duration
        double sumDuration = 0.0;
        double sumOccupancy = 0.0;

        for (const auto& r : records) {
            sumDuration += (double)r.durationNs;
            sumOccupancy += CalculateOccupancy(r);
        }
        stats.durationMeanNs = sumDuration / stats.totalKernels;
        stats.avgTheoreticalOccupancy = sumOccupancy / stats.totalKernels;

        // 2. Calculate Std Dev
        double varianceSum = 0.0;
        for (const auto& r : records) {
            double diff = (double)r.durationNs - stats.durationMeanNs;
            varianceSum += (diff * diff);
        }
        stats.durationStdDevNs = std::sqrt(varianceSum / stats.totalKernels);

        return stats;
    }

private:
    static double CalculateOccupancy(const KernelRecord& k) {
        // Simplified Theoretical Occupancy Logic
        // Real calculation requires specific GPU prop queries (SM count, Max Threads/SM)
        // For GTX 1650 (Turing 7.5):
        // Max Threads per SM = 1024
        // Max Registers per SM = 65536
        // Max Shared Mem per SM = 64KB
        
        const uint32_t MAX_THREADS_PER_SM = 1024;
        const uint32_t MAX_REGS_PER_SM = 65536;
        const uint32_t MAX_SHMEM_PER_SM = 64 * 1024;

        uint32_t threadsPerBlock = k.blockX * k.blockY * k.blockZ;
        if (threadsPerBlock == 0) return 0.0;

        // Limiter 1: Threads
        // This is a rough estimation assuming blocks pack perfectly
        // To do this perfectly we need to know how many blocks fit on an SM
        // But for "Theoretical" estimation per kernel, `threadsPerBlock / MAX` isn't per SM.
        
        // Better approximation: 
        // We calculate "Threads Limited By Registers" and "Threads Limited By SharedMem"
        
        uint32_t limitRegs = k.numRegisters > 0 ? (MAX_REGS_PER_SM / k.numRegisters) : MAX_THREADS_PER_SM;
        uint32_t limitShmem = k.sharedMemoryBytes > 0 ? (MAX_SHMEM_PER_SM / k.sharedMemoryBytes) : MAX_THREADS_PER_SM; // blocks
        
        // This logic is complex without `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
        // We will return a simplified "Thread Utilization" for now as a placeholder
        // which represents (ThreadsInBlock / 1024.0) * 100
        // This shows "Block fullness" rather than "SM Occupancy", but is useful.
        
        double blockFullness = (double)threadsPerBlock / 1024.0;
        if (blockFullness > 1.0) blockFullness = 1.0;
        
        return blockFullness * 100.0;
    }
};
