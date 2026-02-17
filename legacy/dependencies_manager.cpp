#include "dependencies_manager.h"
#include <psapi.h>

namespace fs = std::filesystem;

bool DependenciesManager::PrepareTarget(DWORD pid) {
    std::string targetDir = GetProcessDirectory(pid);
    if (targetDir.empty()) {
        std::cerr << "[DepManager] Failed to get directory for PID " << pid << std::endl;
        return false;
    }

    // 1. Find CUPTI DLL
    std::string cuptiPath = FindCuptiDll();
    if (cuptiPath.empty()) {
        std::cerr << "[DepManager] FATAL: Could not find ANY cupti64_*.dll on this system!" << std::endl;
        return false;
    }

    // 2. Copy to Target Dir
    std::cout << "[DepManager] Deploying " << fs::path(cuptiPath).filename() << " to " << targetDir << "..." << std::endl;
    if (CopyDllToDir(cuptiPath, targetDir)) {
        std::cout << "[DepManager] Success!" << std::endl;
        return true;
    } else {
        std::cerr << "[DepManager] Failed to copy DLL." << std::endl;
        return false;
    }
}

std::string DependenciesManager::FindCuptiDll() {
    // Strategy 1: Check Environment Variable CUDA_PATH
    const char* cudaPath = std::getenv("CUDA_PATH");
    if (cudaPath) {
        std::string searchDir = std::string(cudaPath) + "\\extras\\CUPTI\\lib64";
        if (fs::exists(searchDir)) {
            for (const auto& entry : fs::directory_iterator(searchDir)) {
                std::string filename = entry.path().filename().string();
                if (filename.find("cupti64_") != std::string::npos && filename.find(".dll") != std::string::npos) {
                    return entry.path().string();
                }
            }
        }
        
        // Fallback: Sometimes in bin/ or lib/x64 ?
        searchDir = std::string(cudaPath) + "\\bin";
        // ... (Similar check)
    }

    // Strategy 2: Absolute Bruteforce common paths
    std::vector<std::string> commonPaths = {
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\extras\\CUPTI\\lib64",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\extras\\CUPTI\\lib64",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\extras\\CUPTI\\lib64",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\extras\\CUPTI\\lib64"
    };

    for (const auto& dir : commonPaths) {
        if (fs::exists(dir)) {
            for (const auto& entry : fs::directory_iterator(dir)) {
                 std::string filename = entry.path().filename().string();
                 if (filename.find("cupti64_") != std::string::npos && filename.find(".dll") != std::string::npos) {
                     return entry.path().string();
                 }
            }
        }
    }

    return "";
}

std::string DependenciesManager::GetProcessDirectory(DWORD pid) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (!hProcess) return "";

    char buffer[MAX_PATH];
    if (GetModuleFileNameExA(hProcess, NULL, buffer, MAX_PATH)) {
        std::string fullPath = buffer;
        CloseHandle(hProcess);
        return fs::path(fullPath).parent_path().string();
    }
    
    CloseHandle(hProcess);
    return "";
}

bool DependenciesManager::CopyDllToDir(const std::string& dllPath, const std::string& targetDir) {
    try {
        fs::path source(dllPath);
        fs::path dest = fs::path(targetDir) / source.filename();
        
        if (fs::exists(dest)) {
            std::cout << "[DepManager] DLL already exists at check, skipping copy (File might be in use)." << std::endl;
            return true; 
        }

        BOOL success = CopyFileA(dllPath.c_str(), dest.string().c_str(), FALSE);
        return success != 0;
    } catch (...) {
        return false;
    }
}
