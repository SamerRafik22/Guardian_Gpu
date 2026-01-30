#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>

class CsvLogger {
public:
    CsvLogger() {
        // Create filename with timestamp: gpu_log_YYYYMMDD_HHMMSS.csv
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << "gpu_log_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".csv";
        m_filename = oss.str();

        // Write Header
        std::ofstream file(m_filename, std::ios::out);
        if (file.is_open()) {
            file << "TIMESTAMP,PID,NAME,MEM_MB,PWR_W,GPU_TIME_MS,GPU_PACKET_COUNT,NET_TX,NET_RX\n";
            file.close();
            std::cout << "[INFO] CSV Logger initiated: " << m_filename << std::endl;
        } else {
            std::cerr << "[ERROR] Failed to create CSV file: " << m_filename << std::endl;
        }
    }

    void LogRow(const std::string& timestamp, 
                uint32_t pid, const std::string& procName,
                uint64_t vram, double power, double gpuTimeMs, uint32_t packetCount,
                uint64_t netTx, uint64_t netRx) 
    {
        std::ofstream file(m_filename, std::ios::app);
        if (file.is_open()) {
            file << timestamp << ","
                 << pid << "," << "\"" << procName << "\","
                 << (vram / 1024 / 1024) << ","
                 << std::fixed << std::setprecision(1) << power << ","
                 << std::setprecision(2) << gpuTimeMs << ","
                 << packetCount << ","
                 << netTx << ","
                 << netRx << "\n";
            file.close();
        }
    }

private:
    std::string m_filename;
};
