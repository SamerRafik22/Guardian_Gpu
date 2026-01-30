#include <iostream>
#include <windows.h>

int main() {
    std::cout << "[Target] I am a dummy target. My PID is " << GetCurrentProcessId() << std::endl;
    std::cout << "[Target] Sleeping forever..." << std::endl;
    while(true) {
        Sleep(1000);
    }
    return 0;
}
