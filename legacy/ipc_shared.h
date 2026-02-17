#pragma once
#include <cstdint>

// Named Pipe configuration
const char* const PIPE_NAME = "\\\\.\\pipe\\GuardianPipe";
const uint32_t PIPE_BUFFER_SIZE = 4096;

// The data structure sent from the DLL (Probe) to the EXE (Injector)
// Sent once per second
struct KernelStatsPackage {
    uint32_t processId;
    uint32_t kernelCount;
    double durationMeanNs;
    double durationStdDevNs;
    double avgOccupancy;
    uint32_t memcpyCount;
};
