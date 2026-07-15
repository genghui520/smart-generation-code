#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <Fwlib32.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

using cnc_allclibhndl3_t = decltype(&::cnc_allclibhndl3);
using cnc_freelibhndl_t = decltype(&::cnc_freelibhndl);
using cnc_statinfo_t = decltype(&::cnc_statinfo);

static std::wstring widen(const std::string& text) {
    if (text.empty()) {
        return L"";
    }
    int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (size <= 0) {
        size = MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, nullptr, 0);
        if (size <= 0) {
            return L"";
        }
        std::wstring wide(size - 1, L'\0');
        MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, &wide[0], size);
        return wide;
    }
    std::wstring wide(size - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, &wide[0], size);
    return wide;
}

static std::wstring env_wstring(const wchar_t* name) {
    DWORD needed = GetEnvironmentVariableW(name, nullptr, 0);
    if (needed == 0) {
        return L"";
    }
    std::wstring value(needed - 1, L'\0');
    GetEnvironmentVariableW(name, &value[0], needed);
    return value;
}

static std::wstring join_dll_path(const std::wstring& wide_dir_input) {
    std::wstring wide_dir = wide_dir_input;
    if (wide_dir.empty()) {
        return L"Fwlib32.dll";
    }
    wchar_t last = wide_dir.back();
    if (last != L'\\' && last != L'/') {
        wide_dir += L"\\";
    }
    wide_dir += L"Fwlib32.dll";
    return wide_dir;
}

static std::string win_error_text(DWORD code) {
    char* buffer = nullptr;
    DWORD size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<char*>(&buffer),
        0,
        nullptr);
    std::string message = size && buffer ? std::string(buffer, size) : "";
    if (buffer) {
        LocalFree(buffer);
    }
    return message;
}

static short print_status(const char* label, cnc_statinfo_t cnc_statinfo, unsigned short handle) {
    ODBST status;
    std::memset(&status, 0, sizeof(status));
    short ret = cnc_statinfo ? cnc_statinfo(handle, &status) : -1;
    std::cout << "STATUS " << label
              << " ret=" << ret
              << " tmmode=" << status.tmmode
              << " aut=" << status.aut
              << " run=" << status.run
              << " motion=" << status.motion
              << " mstb=" << status.mstb
              << " emergency=" << status.emergency
              << " alarm=" << status.alarm
              << " edit=" << status.edit
              << "\n";
    return ret;
}

int main(int argc, char** argv) {
    std::wstring dll_dir = env_wstring(L"FOCAS_DLL_DIR");
    bool dll_dir_from_env = !dll_dir.empty();
    if (!dll_dir_from_env && argc > 1) {
        dll_dir = widen(argv[1]);
    }
    int arg_base = dll_dir_from_env ? 1 : 2;
    std::string host = argc > arg_base ? argv[arg_base] : "127.0.0.1";
    int port = argc > arg_base + 1 ? std::atoi(argv[arg_base + 1]) : 8193;
    long timeout_seconds = argc > arg_base + 2 ? std::atol(argv[arg_base + 2]) : 10;

    std::wstring dll_path = join_dll_path(dll_dir);
    if (!dll_dir.empty()) {
        SetDllDirectoryW(dll_dir.c_str());
    }

    std::wcout << L"dll_path=" << dll_path << L"\n";
    std::cout << "host=" << host << "\n";
    std::cout << "port=" << port << "\n";
    std::cout << "timeout_seconds=" << timeout_seconds << "\n";

    HMODULE dll = LoadLibraryW(dll_path.c_str());
    if (!dll) {
        DWORD err = GetLastError();
        std::cout << "LOAD_DLL status=-1 win_error=" << err << " " << win_error_text(err) << "\n";
        return 2;
    }
    std::cout << "LOAD_DLL status=0\n";

    auto cnc_allclibhndl3 = reinterpret_cast<cnc_allclibhndl3_t>(GetProcAddress(dll, "cnc_allclibhndl3"));
    auto cnc_freelibhndl = reinterpret_cast<cnc_freelibhndl_t>(GetProcAddress(dll, "cnc_freelibhndl"));
    auto cnc_statinfo = reinterpret_cast<cnc_statinfo_t>(GetProcAddress(dll, "cnc_statinfo"));
    if (!cnc_allclibhndl3 || !cnc_freelibhndl || !cnc_statinfo) {
        std::cout << "GET_PROC status=-1 cnc_allclibhndl3=" << (cnc_allclibhndl3 != nullptr)
                  << " cnc_freelibhndl=" << (cnc_freelibhndl != nullptr)
                  << " cnc_statinfo=" << (cnc_statinfo != nullptr) << "\n";
        FreeLibrary(dll);
        return 3;
    }
    std::cout << "GET_PROC status=0\n";

    unsigned short handle = 0;
    short ret = cnc_allclibhndl3(host.c_str(), static_cast<unsigned short>(port), timeout_seconds, &handle);
    std::cout << "CONNECT ret=" << ret << " handle=" << handle << "\n";
    if (ret == 0) {
        print_status("current", cnc_statinfo, handle);
        short free_ret = cnc_freelibhndl(handle);
        std::cout << "DISCONNECT ret=" << free_ret << "\n";
    }

    FreeLibrary(dll);
    return ret == 0 ? 0 : 1;
}
