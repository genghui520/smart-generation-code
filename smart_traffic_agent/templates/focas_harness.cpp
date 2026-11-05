#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "Fwlib32.h"
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

struct FocasApi {
    HMODULE dll = nullptr;
__FOCAS_FIELDS__

    bool load()
    {
        std::wstring dll_path = L"Fwlib32.dll";
        wchar_t dll_dir[32768]{};
        DWORD dll_dir_length = GetEnvironmentVariableW(L"FOCAS_DLL_DIR", dll_dir, 32768);
        if (dll_dir_length > 0 && dll_dir_length < 32768) {
            std::wstring directory(dll_dir, dll_dir_length);
            if (!directory.empty() && directory.back() != L'\\' && directory.back() != L'/') {
                directory += L'\\';
            }
            dll_path = directory + L"Fwlib32.dll";
        }
        dll = LoadLibraryW(dll_path.c_str());
        if (!dll) return false;
__FOCAS_BINDINGS__
        return true;
    }

    ~FocasApi()
    {
        if (dll) FreeLibrary(dll);
    }
};

static bool check_rc(short ret, const char* name)
{
    if (ret == EW_OK) return true;
    std::cerr << name << " failed, ret=" << ret << '\n';
    return false;
}

// The deterministic API recipes share the production ``ctx.call`` boundary.
// Code-only evaluation has no PCAP/CSV context, so this adapter executes the
// supplied SDK lambda and intentionally leaves capture side effects disabled.
class CaptureContext {
public:
    template <class Fn, class Invoke, class Serialize>
    short call(const std::string&, const std::string&, const std::string&, Fn,
               Invoke invoke, Serialize) {
        return invoke();
    }

    template <class... Args>
    void block(Args&&...) {}

    template <class... Args>
    void host(Args&&...) {}
};

#include "scenario_fragment.inc"

int main()
{
    FocasApi api;
    unsigned short handle = 0;
    bool connected = false;
    try {
        if (!api.load()) {
            std::cerr << "FOCAS DLL load failed\n";
            return 1;
        }

        const char* host = std::getenv("FOCAS_HOST");
        const char* port_env = std::getenv("FOCAS_PORT");
        int port = port_env ? std::stoi(port_env) : 8193;
        if (port < 1 || port > 65535) throw std::runtime_error("Invalid FOCAS_PORT");
        short ret = api.cnc_allclibhndl3(host && *host ? host : "127.0.0.1", static_cast<unsigned short>(port), 10, &handle);
        if (ret != EW_OK) {
            std::cerr << "FOCAS connect failed, ret=" << ret << '\n';
            return 1;
        }
        connected = true;

        ret = run_generated_scenario(api, handle);
        if (connected) api.cnc_freelibhndl(handle);
        connected = false;
        if (ret != EW_OK) {
            std::cerr << "Scenario failed, ret=" << ret << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Scenario exception: " << exc.what() << '\n';
    } catch (...) {
        std::cerr << "Scenario exception: unknown\n";
    }
    if (connected) api.cnc_freelibhndl(handle);
    return 1;
}
