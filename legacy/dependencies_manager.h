#pragma once
#include <string>
#include <windows.h>
#include <iostream>
#include <filesystem>
#include <vector>

class DependenciesManager {
public:
    static bool PrepareTarget(DWORD pid);

private:
    static std::string FindCuptiDll();
    static std::string GetProcessDirectory(DWORD pid);
    static bool CopyDllToDir(const std::string& dllPath, const std::string& targetDir);
};
