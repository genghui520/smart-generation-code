#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <Windows.h>
#include "Fwlib32.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <utility>

// Production infrastructure is owned by SMPAgent, never by the generated fragment.
struct pcap;
struct pcap_dumper;
struct pcap_pkthdr { timeval ts; unsigned int caplen; unsigned int len; };
using pcap_handler = void(__cdecl *)(unsigned char*, const pcap_pkthdr*, const unsigned char*);

static std::string env(const char* name, const char* fallback = "") {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}
static long long timestamp_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
static std::string timestamp() {
    SYSTEMTIME now{};
    GetSystemTime(&now);
    char text[40]{};
    sprintf_s(text, "%04u-%02u-%02uT%02u:%02u:%02u.%03uZ", now.wYear, now.wMonth,
              now.wDay, now.wHour, now.wMinute, now.wSecond, now.wMilliseconds);
    return text;
}
static std::string csv(const std::string& text) {
    std::string result = "\"";
    for (char c : text) result += c == '"' ? "\"\"" : std::string(1, c);
    return result + "\"";
}
static std::string RetText(short ret) {
    switch (ret) {
__RETURN_CODE_CASES__
    default: return "FOCAS_RETURN_" + std::to_string(ret);
    }
}

// Keep every cnc_statinfo record comparable across preflight, diagnostics,
// operator polling, and postcheck paths. Values are read directly from the
// returned ODBST; this function only fixes the serialized field order.
static std::string StatinfoData(const ODBST& status) {
    return "aut=" + std::to_string(status.aut) + ";run=" + std::to_string(status.run) +
        ";motion=" + std::to_string(status.motion) + ";emergency=" +
        std::to_string(status.emergency) + ";alarm=" + std::to_string(status.alarm) +
        ";edit=" + std::to_string(status.edit);
}

class PacketSniffer {
public:
    PacketSniffer(const std::string& device, const std::string& file,
                  const std::string& api_directory, const std::string& batch_id)
        : device_(device), file_(file), api_directory_(api_directory), batch_id_(batch_id) {}
    ~PacketSniffer() { Stop(); }
    bool Start() {
        char windows[MAX_PATH]{};
        GetWindowsDirectoryA(windows, MAX_PATH);
        std::string path = std::string(windows) + "\\SysWOW64\\Npcap\\wpcap.dll";
        dll_ = LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!dll_) dll_ = LoadLibraryA("wpcap.dll");
        if (!dll_) { error_ = "LoadLibrary(wpcap.dll) failed"; return false; }
        open_ = reinterpret_cast<decltype(open_)>(GetProcAddress(dll_, "pcap_open_live"));
        nonblock_ = reinterpret_cast<decltype(nonblock_)>(GetProcAddress(dll_, "pcap_setnonblock"));
        dump_open_ = reinterpret_cast<decltype(dump_open_)>(GetProcAddress(dll_, "pcap_dump_open"));
        dispatch_ = reinterpret_cast<decltype(dispatch_)>(GetProcAddress(dll_, "pcap_dispatch"));
        dump_ = reinterpret_cast<decltype(dump_)>(GetProcAddress(dll_, "pcap_dump"));
        flush_ = reinterpret_cast<decltype(flush_)>(GetProcAddress(dll_, "pcap_dump_flush"));
        dump_close_ = reinterpret_cast<decltype(dump_close_)>(GetProcAddress(dll_, "pcap_dump_close"));
        close_ = reinterpret_cast<decltype(close_)>(GetProcAddress(dll_, "pcap_close"));
        if (!open_ || !nonblock_ || !dump_open_ || !dispatch_ || !dump_ || !flush_ || !dump_close_ || !close_) {
            error_ = "Npcap required export missing"; return false;
        }
        char errbuf[512]{};
        handle_ = open_(device_.c_str(), 65535, 0, 1, errbuf);
        if (!handle_) { error_ = errbuf; return false; }
        if (nonblock_(handle_, 1, errbuf) != 0) { error_ = errbuf; return false; }
        dumper_ = dump_open_(handle_, file_.c_str());
        if (!dumper_) { error_ = "pcap_dump_open failed"; return false; }
        running_ = true;
        thread_ = std::thread([this] {
            while (running_) {
                int count = dispatch_(handle_, 256, &PacketSniffer::Dump, reinterpret_cast<unsigned char*>(this));
                if (count < 0 || flush_(dumper_) != 0) { failed_ = true; break; }
                if (count == 0) Sleep(2);
            }
            // Drain buffered packets before closing the file.
            for (int drain = 0; drain < 64; ++drain) {
                int count = dispatch_(handle_, 256, &PacketSniffer::Dump, reinterpret_cast<unsigned char*>(this));
                if (count < 0) failed_ = true;
                if (count <= 0) break;
            }
            if (flush_(dumper_) != 0) failed_ = true;
        });
        return true;
    }
    bool BeginApi(const std::string& api, long long start_ns) {
        if (!handle_ || !dump_open_) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = api_dumpers_.find(api);
        if (found == api_dumpers_.end()) {
            std::string safe = api;
            for (char& c : safe) {
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) c = '_';
            }
            std::string safe_batch = batch_id_;
            for (char& c : safe_batch) {
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) c = '_';
            }
            std::string path = api_directory_ + "\\" + safe_batch + "_" + safe + ".pcap";
            pcap_dumper* api_dumper = dump_open_(handle_, path.c_str());
            if (!api_dumper) { error_ = "pcap_dump_open failed for API " + api; failed_ = true; return false; }
            found = api_dumpers_.emplace(api, api_dumper).first;
        }
        // Attribution is based on the packet timestamp, not on which API the
        // background thread happens to be processing when the packet arrives.
        // This prevents a short call's response from being assigned to the
        // following helper call.
        api_windows_.push_back(ApiWindow{api, start_ns, LLONG_MAX});
        return true;
    }
    void EndApi(const std::string& api, long long end_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = api_dumpers_.find(api);
        if (found != api_dumpers_.end() && flush_) flush_(found->second);
        for (auto it = api_windows_.rbegin(); it != api_windows_.rend(); ++it) {
            if (it->api == api && it->end_ns == LLONG_MAX) {
                it->end_ns = end_ns;
                break;
            }
        }
    }
    // Capture runs continuously on the background thread. Keep this hook so
    // the invocation wrapper can use the same lifecycle as the synchronous
    // fixed runner without opening a second Npcap handle.
    void CapturePackets(int wait_ms = 4) {
        // The capture loop is asynchronous. Keep the API interval open for a
        // bounded drain period so packets queued by Npcap after a very fast
        // FOCAS return are still attributed to the correct invocation.
        if (wait_ms > 0) Sleep(static_cast<DWORD>(wait_ms));
    }
    void Stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& item : api_dumpers_) {
            if (flush_) flush_(item.second);
            if (dump_close_) dump_close_(item.second);
        }
        api_dumpers_.clear();
        api_windows_.clear();
        if (dumper_) { dump_close_(dumper_); dumper_ = nullptr; }
        if (handle_) { close_(handle_); handle_ = nullptr; }
        if (dll_) { FreeLibrary(dll_); dll_ = nullptr; }
    }
    bool Failed() const { return failed_; }
    std::string Error() const { return error_; }
private:
    struct ApiWindow {
        std::string api;
        long long start_ns;
        long long end_ns;
    };
    static long long PacketTimestampNs(const pcap_pkthdr* header) {
        if (!header) return 0;
        return static_cast<long long>(header->ts.tv_sec) * 1000000000LL +
               static_cast<long long>(header->ts.tv_usec) * 1000LL;
    }
    static void __cdecl Dump(unsigned char* user, const pcap_pkthdr* header, const unsigned char* packet) {
        auto* self = reinterpret_cast<PacketSniffer*>(user);
        std::lock_guard<std::mutex> lock(self->mutex_);
        if (self->dumper_) self->dump_(reinterpret_cast<unsigned char*>(self->dumper_), header, packet);
        const long long packet_ns = PacketTimestampNs(header);
        // A packet belongs to at most one invocation interval. Windows are
        // sequential because CaptureContext closes one before opening the
        // next, so this does not duplicate traffic between APIs.
        for (const auto& window : self->api_windows_) {
            if (packet_ns < window.start_ns || packet_ns > window.end_ns) continue;
            auto found = self->api_dumpers_.find(window.api);
            if (found != self->api_dumpers_.end()) {
                self->dump_(reinterpret_cast<unsigned char*>(found->second), header, packet);
            }
            break;
        }
    }
    std::string device_, file_, api_directory_, batch_id_, error_;
    HMODULE dll_ = nullptr;
    pcap* handle_ = nullptr;
    pcap_dumper* dumper_ = nullptr;
    std::map<std::string, pcap_dumper*> api_dumpers_;
    std::vector<ApiWindow> api_windows_;
    std::mutex mutex_;
    std::atomic<bool> running_{false}, failed_{false};
    std::thread thread_;
    pcap* (__cdecl *open_)(const char*, int, int, int, char*) = nullptr;
    int (__cdecl *nonblock_)(pcap*, int, char*) = nullptr;
    pcap_dumper* (__cdecl *dump_open_)(pcap*, const char*) = nullptr;
    int (__cdecl *dispatch_)(pcap*, int, pcap_handler, unsigned char*) = nullptr;
    void (__cdecl *dump_)(unsigned char*, const pcap_pkthdr*, const unsigned char*) = nullptr;
    int (__cdecl *flush_)(pcap_dumper*) = nullptr;
    void (__cdecl *dump_close_)(pcap_dumper*) = nullptr;
    void (__cdecl *close_)(pcap*) = nullptr;
};

struct FocasApi {
    HMODULE dll = nullptr;
__FOCAS_FIELDS__
    bool load() {
        wchar_t directory[32768]{};
        DWORD count = GetEnvironmentVariableW(L"FOCAS_DLL_DIR", directory, 32768);
        std::wstring path = L"Fwlib32.dll";
        if (count && count < 32768) {
            SetDllDirectoryW(directory);
            path = std::wstring(directory) + L"\\Fwlib32.dll";
        }
        dll = LoadLibraryW(path.c_str());
        if (!dll) return false;
__FOCAS_BINDINGS__
        return cnc_allclibhndl3 && cnc_freelibhndl;
    }
    ~FocasApi() { if (dll) FreeLibrary(dll); }
};

class CaptureContext {
public:
    explicit CaptureContext(PacketSniffer* sniffer = nullptr) : sniffer_(sniffer),
                       input_("data\\focas_api_input.csv", std::ios::binary),
                       output_("data\\focas_api_output.csv", std::ios::binary) {
        if (!input_ || !output_) throw std::runtime_error("Cannot create API CSV logs");
        input_ << "\xEF\xBB\xBFindex,timestamp,step_id,interface_name,protocol_function,parameters,invocation_id,transaction_id,start_timestamp_ns,device_context\n";
        output_ << "\xEF\xBB\xBFindex,timestamp,step_id,api_name,return_code,return_text,data,invocation_id,transaction_id,end_timestamp_ns\n";
        nonce_ = std::to_string(GetCurrentProcessId()) + "-" + std::to_string(timestamp_ns());
    }
    template<class Function, class Callable, class Output>
    short call(const std::string& step, const std::string& name, const std::string& parameters,
               Function symbol, Callable invoke, Output build_output) {
        if (!symbol) { block(name, "export_missing"); return -15; }
        const int index = ++index_;
        const std::string id = "inv-" + nonce_ + "-" + std::to_string(index);
        const std::string tx = "tx-" + nonce_ + "-" + std::to_string(index);
        long long start = timestamp_ns();
        if (sniffer_) sniffer_->BeginApi(name, start);
        // Keep the complete runtime argument payload.  The fixed log contract
        // counts FlibHndl because it is a real parameter in the SDK
        // signature; the legacy dataset projection may still filter it later.
        write_input(index, step, name, parameters, id, tx, start);
        short ret = EW_OK;
        try { ret = invoke(); }
        catch (...) {
            if (sniffer_) sniffer_->EndApi(name, timestamp_ns());
            throw;
        }
        if (sniffer_) {
            sniffer_->CapturePackets();
        }
        long long end = timestamp_ns();
        if (sniffer_) sniffer_->EndApi(name, end);
        // Failed outputs are undefined and must never be interpreted as device values.
        std::string data = ret == EW_OK ? build_output(ret) : "output_valid=false";
        write_output(index, step, name, ret, RetText(ret), data, id, tx, end);
        std::cout << name << " -> ret=" << ret << " " << RetText(ret) << "; " << data << std::endl;
        if (ret != EW_OK) diagnose(true);
        return ret;
    }
    // A step is a log label, not a FOCAS argument. Accept legacy numeric labels
    // through the same implementation so invocation timing and IDs stay identical.
    template<class Function, class Callable, class Output>
    short call(int step, const std::string& name, const std::string& parameters,
               Function symbol, Callable invoke, Output build_output) {
        return call(std::to_string(step), name, parameters, symbol, invoke, build_output);
    }
    void block(const std::string& api, const std::string& reason) {
        host("host_api_prerequisite", 12, "API_PREREQUISITE_BLOCKED", "function=" + api + ";reason=" + reason);
        failed_ = true;
        diagnose(false);
    }
    bool wait_for_operator(FocasApi& api, unsigned short handle, const std::string& action,
                           const std::string& target, int timeout_seconds = 60) {
        const std::string safe_action = action == "remote" ? "remote" : "reset";
        if (env("SMP_SKIP_MANUAL_ACTION_PROMPTS", "0") == "1") {
            host("operator_action", 12, "MANUAL_ACTION_DEFERRED",
                "action=" + safe_action + ";target=" + target + ";reason=earlier_wait_timed_out");
            return false;
        }
        std::cout << "@@SMP@@{\"event\":\"operator_action_required\",\"data\":{\"action\":\""
                  << safe_action << "\",\"target\":\"" << target
                  << "\",\"timeout_seconds\":" << timeout_seconds << "}}" << std::endl;
        // An operator action is an explicit prerequisite. Keep the process
        // alive until FOCAS observes the requested state; the desktop task
        // can be cancelled by the operator when the action is not possible.
        const bool wait_indefinitely = true;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
        int attempt = 0;
        while (wait_indefinitely || std::chrono::steady_clock::now() < deadline) {
            ODBST state{};
            short ret = call("OPERATOR_STATE_POLL", "cnc_statinfo",
                "FlibHndl=" + std::to_string(handle) + ";action=" + safe_action +
                    ";attempt=" + std::to_string(++attempt), api.cnc_statinfo,
                [&] { return api.cnc_statinfo(handle, &state); }, [&](short) {
                    return StatinfoData(state);
                });
            bool ready = ret == EW_OK && state.alarm == 0 && state.emergency == 0 &&
                state.motion == 0 && (state.run == 0 || state.run == 1);
            if (safe_action == "remote") ready = ready && state.aut == 10;
            if (safe_action == "reset" && ret == EW_OK && api.cnc_alarm2) {
                long alarm_bits = 0;
                short alarm_ret = call("OPERATOR_STATE_POLL", "cnc_alarm2",
                    "FlibHndl=" + std::to_string(handle) + ";attempt=" + std::to_string(attempt),
                    api.cnc_alarm2, [&] { return api.cnc_alarm2(handle, &alarm_bits); },
                    [&](short) { return "alarm_bits=" + std::to_string(alarm_bits); });
                ready = ready && alarm_ret == EW_OK && alarm_bits == 0;
            }
            if (ready) {
                host("operator_action", 0, "MANUAL_ACTION_VERIFIED",
                    "action=" + safe_action + ";target=" + target + ";aut=" +
                    std::to_string(state.aut) + ";alarm=" + std::to_string(state.alarm) +
                    ";emergency=" + std::to_string(state.emergency));
                std::cout << "@@SMP@@{\"event\":\"operator_action_verified\",\"data\":{\"action\":\""
                          << safe_action << "\",\"target\":\"" << target << "\"}}" << std::endl;
                return true;
            }
            Sleep(1000);
        }
        host("operator_action", 12, "MANUAL_ACTION_TIMEOUT",
            "action=" + safe_action + ";target=" + target + ";timeout_seconds=" +
            std::to_string(timeout_seconds));
        manual_action_timed_out_ = true;
        std::cout << "@@SMP@@{\"event\":\"operator_action_timeout\",\"data\":{\"action\":\""
                  << safe_action << "\",\"target\":\"" << target << "\"}}" << std::endl;
        return false;
    }
    bool wait_for_cnc_mode(FocasApi& api, unsigned short handle, short expected_aut,
                          const std::string& required_mode, const std::string& target,
                          int timeout_seconds = 60) {
        if (expected_aut < 0) {
            host("operator_action", 12, "MODE_EVIDENCE_REQUIRED",
                "target=" + target + ";required_mode=" + required_mode + ";expected_aut=unknown");
            return false;
        }
        std::cout << "@@SMP@@{\"event\":\"operator_action_required\",\"data\":{\"action\":\"mode\",\"target\":\""
                  << target << "\",\"required_mode\":\"" << required_mode
                  << "\",\"timeout_seconds\":" << timeout_seconds << "}}" << std::endl;
        const bool wait_indefinitely = true;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
        int attempt = 0;
        while (wait_indefinitely || std::chrono::steady_clock::now() < deadline) {
            ODBST state{};
            short ret = call("OPERATOR_MODE_POLL", "cnc_statinfo",
                "FlibHndl=" + std::to_string(handle) + ";required_mode=" + required_mode +
                    ";expected_aut=" + std::to_string(expected_aut) + ";attempt=" + std::to_string(++attempt),
                api.cnc_statinfo, [&] { return api.cnc_statinfo(handle, &state); }, [&](short) {
                    return StatinfoData(state);
                });
            const bool ready = ret == EW_OK && state.aut == expected_aut && state.alarm == 0 &&
                state.emergency == 0 && state.motion == 0 && (state.run == 0 || state.run == 1);
            if (ready) {
                host("operator_action", 0, "MODE_REQUIREMENT_VERIFIED",
                    "target=" + target + ";required_mode=" + required_mode +
                        ";aut=" + std::to_string(state.aut));
                std::cout << "@@SMP@@{\"event\":\"operator_action_verified\",\"data\":{\"action\":\"mode\",\"target\":\""
                          << target << "\",\"required_mode\":\"" << required_mode << "\"}}" << std::endl;
                return true;
            }
            Sleep(1000);
        }
        host("operator_action", 12, "MODE_REQUIREMENT_TIMEOUT",
            "target=" + target + ";required_mode=" + required_mode +
                ";expected_aut=" + std::to_string(expected_aut));
        manual_action_timed_out_ = true;
        std::cout << "@@SMP@@{\"event\":\"operator_action_timeout\",\"data\":{\"action\":\"mode\",\"target\":\""
                  << target << "\",\"required_mode\":\"" << required_mode << "\"}}" << std::endl;
        return false;
    }
    bool manual_action_timed_out() const { return manual_action_timed_out_; }
    void set_error_diagnostics(std::function<void(bool)> callback) { error_diagnostics_ = std::move(callback); }
    void diagnose(bool include_error_detail = false) {
        if (!error_diagnostics_ || diagnosing_) return;
        diagnosing_ = true;
        try { error_diagnostics_(include_error_detail); } catch (...) {}
        diagnosing_ = false;
    }
    void host(const std::string& name, short ret, const std::string& label, const std::string& data) {
        int index = ++index_;
        std::string id = "inv-" + nonce_ + "-" + std::to_string(index);
        write_input(index, name, name, "", id, "", timestamp_ns());
        write_output(index, name, name, ret, label, data, id, "", timestamp_ns());
    }
    bool failed() const { return failed_; }
private:
    static std::string parameters_without_handle(const std::string& parameters) {
        std::stringstream source(parameters);
        std::string part;
        std::string result;
        while (std::getline(source, part, ';')) {
            const auto equals = part.find('=');
            std::string key = equals == std::string::npos ? part : part.substr(0, equals);
            while (!key.empty() && std::isspace(static_cast<unsigned char>(key.front()))) key.erase(key.begin());
            while (!key.empty() && std::isspace(static_cast<unsigned char>(key.back()))) key.pop_back();
            std::string folded = key;
            std::transform(folded.begin(), folded.end(), folded.begin(),
                           [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
            if (folded == "flibhndl") continue;
            if (!result.empty() && !part.empty()) result += ";";
            result += part;
        }
        return result;
    }

    void write_input(int index, const std::string& step, const std::string& api, const std::string& params,
                     const std::string& id, const std::string& tx, long long ns) {
        std::string device = "target=NCGuide;endpoint=" + env("FOCAS_HOST", "127.0.0.1") + ":" + env("FOCAS_PORT", "8193");
        input_ << index << ',' << csv(timestamp()) << ',' << csv(step) << ',' << csv(api) << ',' << csv(api)
               << ',' << csv(params) << ',' << csv(id) << ',' << csv(tx)
               << ',' << ns << ',' << csv(device) << '\n';
        input_.flush();
    }
    void write_output(int index, const std::string& step, const std::string& api, short ret,
                      const std::string& label, const std::string& data, const std::string& id,
                      const std::string& tx, long long ns) {
        output_ << index << ',' << csv(timestamp()) << ',' << csv(step) << ',' << csv(api) << ',' << ret
                << ',' << csv(label) << ',' << csv(data) << ',' << csv(id)
                << ',' << csv(tx) << ',' << ns << '\n';
        output_.flush();
    }
    int index_ = 0;
    bool failed_ = false;
    bool diagnosing_ = false;
    bool manual_action_timed_out_ = false;
    std::string nonce_;
    std::function<void(bool)> error_diagnostics_;
    PacketSniffer* sniffer_ = nullptr;
    std::ofstream input_, output_;
};

static bool capture_alarm_diagnostics(FocasApi& api, unsigned short handle, CaptureContext& ctx,
                                      bool* sw100 = nullptr) {
    if (sw100) *sw100 = false;
    ODBST status{};
    const short status_rc = api.cnc_statinfo
        ? ctx.call("ALARM_DIAGNOSTIC", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
            [&] { return api.cnc_statinfo(handle, &status); }, [&](short) {
                return StatinfoData(status);
            })
        : -15;
    if (!api.cnc_statinfo) ctx.host("cnc_statinfo", -15, "UNSUPPORTED_SYMBOL", "alarm status reader is not exported");

    ODBALM classic{};
    const short classic_rc = api.cnc_alarm
        ? ctx.call("ALARM_DIAGNOSTIC", "cnc_alarm", "FlibHndl=" + std::to_string(handle), api.cnc_alarm,
            [&] { return api.cnc_alarm(handle, &classic); },
            [&](short) { return "alarm_status=" + std::to_string(classic.data); })
        : -15;
    if (!api.cnc_alarm) ctx.host("cnc_alarm", -15, "UNSUPPORTED_SYMBOL", "alarm status reader is not exported");

    long alarm_bits = 0;
    const short bits_rc = api.cnc_alarm2
        ? ctx.call("ALARM_DIAGNOSTIC", "cnc_alarm2", "FlibHndl=" + std::to_string(handle), api.cnc_alarm2,
            [&] { return api.cnc_alarm2(handle, &alarm_bits); },
            [&](short) { return "alarm_bits=" + std::to_string(alarm_bits); })
        : -15;
    if (!api.cnc_alarm2) ctx.host("cnc_alarm2", -15, "UNSUPPORTED_SYMBOL", "alarm status reader is not exported");

    const bool alarm_present = (status_rc == EW_OK && status.alarm != 0) ||
        (classic_rc == EW_OK && classic.data != 0) || (bits_rc == EW_OK && alarm_bits != 0);
    if (!alarm_present) return false;

    constexpr short message_capacity = 16;
    short message_count = message_capacity;
    ODBALMMSG2 messages[message_capacity]{};
    if (api.cnc_rdalmmsg2) {
        short message_rc = ctx.call("ALARM_DIAGNOSTIC", "cnc_rdalmmsg2",
            "FlibHndl=" + std::to_string(handle) + ";type=-1;num=" + std::to_string(message_count),
            api.cnc_rdalmmsg2,
            [&] { return api.cnc_rdalmmsg2(handle, -1, &message_count, messages); }, [&](short) {
                std::string output = "num=" + std::to_string(message_count);
                const short safe_count = std::max<short>(0, std::min<short>(message_count, message_capacity));
                for (short i = 0; i < safe_count; ++i) {
                    const auto& alarm = messages[i];
                    const std::string prefix = ";alarm[" + std::to_string(i) + "].";
                    output += prefix + "number=" + std::to_string(alarm.alm_no);
                    output += prefix + "type=" + std::to_string(alarm.type);
                    output += prefix + "axis=" + std::to_string(alarm.axis);
                    output += prefix + "message=" + std::string(alarm.alm_msg,
                        alarm.alm_msg + std::max<short>(0, std::min<short>(alarm.msg_len, sizeof(alarm.alm_msg))));
                }
                return output;
            });
        if (sw100 && message_rc == EW_OK && bits_rc == EW_OK && (alarm_bits & 1L)) {
            const short safe_count = std::max<short>(0, std::min<short>(message_count, message_capacity));
            for (short i = 0; i < safe_count; ++i)
                if (messages[i].alm_no == 100) *sw100 = true;
        }
    } else {
        ctx.host("cnc_rdalmmsg2", -15, "UNSUPPORTED_SYMBOL", "active alarm message reader is not exported");
    }

    if (!api.cnc_stopophis || !api.cnc_startophis || !api.cnc_rdalmhisno || !api.cnc_rdalmhistry5) {
        ctx.host("alarm_history", -15, "UNSUPPORTED_SYMBOL", "required alarm history lifecycle exports are incomplete");
        return true;
    }
    short history_rc = ctx.call("ALARM_HISTORY", "cnc_stopophis",
        "FlibHndl=" + std::to_string(handle), api.cnc_stopophis,
        [&] { return api.cnc_stopophis(handle); }, [](short) { return "sampling_stopped=true"; });
    if (history_rc != EW_OK) return true;

    unsigned short history_count = 0;
    history_rc = ctx.call("ALARM_HISTORY", "cnc_rdalmhisno",
        "FlibHndl=" + std::to_string(handle), api.cnc_rdalmhisno,
        [&] { return api.cnc_rdalmhisno(handle, &history_count); },
        [&](short) { return "history_count=" + std::to_string(history_count); });
    if (history_rc == EW_OK && history_count > 0) {
        const unsigned short end_record = history_count;
        const unsigned short start_record = static_cast<unsigned short>(history_count > 10 ? history_count - 9 : 1);
        ODBAHIS5 history{};
        const unsigned short length = static_cast<unsigned short>(sizeof(history));
        ctx.call("ALARM_HISTORY", "cnc_rdalmhistry5",
            "FlibHndl=" + std::to_string(handle) + ";s_no=" + std::to_string(start_record) +
                ";e_no=" + std::to_string(end_record) + ";length=" + std::to_string(length),
            api.cnc_rdalmhistry5,
            [&] { return api.cnc_rdalmhistry5(handle, start_record, end_record, length, &history); }, [&](short) {
                std::string output = "s_no=" + std::to_string(history.s_no) + ";e_no=" + std::to_string(history.e_no);
                const int records = std::min<int>(10, end_record - start_record + 1);
                for (int i = 0; i < records; ++i) {
                    const auto& alarm = history.alm_his[i];
                    const std::string prefix = ";history[" + std::to_string(i) + "].";
                    output += prefix + "type=" + std::to_string(alarm.alm_grp);
                    output += prefix + "number=" + std::to_string(alarm.alm_no);
                    output += prefix + "axis=" + std::to_string(alarm.axis_no);
                    output += prefix + "path=" + std::to_string(alarm.pth_no);
                    output += prefix + "time=" + std::to_string(alarm.year) + "-" +
                        std::to_string(alarm.month) + "-" + std::to_string(alarm.day) + "T" +
                        std::to_string(alarm.hour) + ":" + std::to_string(alarm.minute) + ":" +
                        std::to_string(alarm.second);
                    output += prefix + "message=" + std::string(alarm.alm_msg,
                        alarm.alm_msg + std::max<short>(0, std::min<short>(alarm.len_msg, sizeof(alarm.alm_msg))));
                }
                return output;
            });
    }
    ctx.call("ALARM_HISTORY_RESTORE", "cnc_startophis",
        "FlibHndl=" + std::to_string(handle), api.cnc_startophis,
        [&] { return api.cnc_startophis(handle); }, [](short) { return "sampling_restarted=true"; });
    return true;
}

static bool recover_alarm(FocasApi& api, unsigned short handle, CaptureContext& ctx) {
    bool sw100 = false;
    capture_alarm_diagnostics(api, handle, ctx, &sw100);
    if (!api.cnc_statinfo || !api.cnc_alarm2) {
        ctx.host("alarm_recovery", -15, "ALARM_RECOVERY_BLOCKED", "reason=verification_reader_missing");
        return false;
    }
    ODBST state{};
    long alarm_bits = 0;
    auto read_state = [&]() {
        short status_rc = ctx.call("ALARM_RECOVERY_VERIFY", "cnc_statinfo", "FlibHndl=" + std::to_string(handle),
            api.cnc_statinfo, [&] { return api.cnc_statinfo(handle, &state); }, [&](short) {
                return StatinfoData(state);
            });
        if (status_rc != EW_OK) return status_rc;
        return ctx.call("ALARM_RECOVERY_VERIFY", "cnc_alarm2", "FlibHndl=" + std::to_string(handle),
            api.cnc_alarm2, [&] { return api.cnc_alarm2(handle, &alarm_bits); },
            [&](short) { return "alarm_bits=" + std::to_string(alarm_bits); });
    };
    if (read_state() != EW_OK) return false;
    if (state.emergency != 0) {
        if (!ctx.wait_for_operator(api, handle, "reset", "alarm_recovery")) return false;
        if (read_state() != EW_OK) return false;
    }
    if (state.motion != 0 || (state.run != 0 && state.run != 1)) {
        ctx.host("alarm_recovery", 12, "ALARM_RECOVERY_BLOCKED", "reason=controller_not_stopped");
        return false;
    }
    if (sw100 && (state.alarm != 0 || alarm_bits != 0)) {
        if (!api.cnc_clralm) {
            ctx.host("cnc_clralm", -15, "UNSUPPORTED_SYMBOL", "id=0;reason=export_missing");
        } else {
            ctx.call("ALARM_RECOVERY", "cnc_clralm", "FlibHndl=" + std::to_string(handle) + ";id=0",
                api.cnc_clralm, [&] { return api.cnc_clralm(handle, 0); },
                [](short) { return "clear_requested=true;id=0"; });
            Sleep(100);
            if (read_state() != EW_OK) return false;
        }
    }
    if (state.alarm != 0 || alarm_bits != 0) {
        if (!api.cnc_reset) {
            ctx.host("cnc_reset", -15, "UNSUPPORTED_SYMBOL", "reason=export_missing");
            return false;
        }
        ctx.call("ALARM_RECOVERY", "cnc_reset", "FlibHndl=" + std::to_string(handle),
            api.cnc_reset, [&] { return api.cnc_reset(handle); },
            [](short) { return "reset_requested=true"; });
        for (int attempt = 0; attempt < 5; ++attempt) {
            Sleep(100);
            if (read_state() != EW_OK) return false;
            if (state.alarm == 0 && alarm_bits == 0) break;
        }
    }
    if (state.alarm != 0 || alarm_bits != 0 || state.emergency != 0) {
        if (!ctx.wait_for_operator(api, handle, "reset", "alarm_recovery")) return false;
        if (read_state() != EW_OK) return false;
    }
    const bool ready = state.alarm == 0 && alarm_bits == 0 && state.emergency == 0 && state.motion == 0 &&
        (state.run == 0 || state.run == 1);
    const bool deferred_for_operator = !ready && ctx.manual_action_timed_out();
    ctx.host("alarm_recovery", (ready || deferred_for_operator) ? 0 : 12,
        ready ? "ALARM_RECOVERY_VERIFIED" :
            (deferred_for_operator ? "ALARM_RECOVERY_DEFERRED_READ_ONLY" : "ALARM_RECOVERY_BLOCKED"),
        "alarm=" + std::to_string(state.alarm) + ";alarm_bits=" + std::to_string(alarm_bits) +
        ";emergency=" + std::to_string(state.emergency) +
        ";run=" + std::to_string(state.run) + ";motion=" + std::to_string(state.motion));
    return ready;
}

#include "scenario_fragment.inc"

__EXECUTION_DRIVER__

int main() {
    CreateDirectoryW(L"data", nullptr);
    CreateDirectoryW(L"data\\by_api", nullptr);
    {
        std::ofstream marker("data\\by_api\\.runtime_per_api_capture", std::ios::binary);
        marker << "mode=runtime_per_api_dumper;batch_id="
               << env("SMPAGENT_BATCH_ID", "batch_unknown") << "\n";
    }
    PacketSniffer sniffer(env("PCAP_NETWORK_DEVICE", "\\Device\\NPF_Loopback"),
                          "data\\focas_capture.pcap", "data\\by_api",
                          env("SMPAGENT_BATCH_ID", "batch_unknown"));
    CaptureContext ctx(&sniffer);
    if (!sniffer.Start()) {
        ctx.host("pcap_open_live", -1, "PCAP_CAPTURE_FAILED", sniffer.Error());
        std::cerr << "PCAP_CAPTURE_FAILED: " << sniffer.Error() << std::endl;
        return 2;
    }
    ctx.host("pcap_open_live", 0, "PCAP_CAPTURE_STARTED", "pcap_capture_enabled=true;pcap_file=data\\focas_capture.pcap");
    FocasApi api;
    unsigned short handle = 0;
    bool connected = false;
    int result = 0;
    try {
        if (!api.load()) throw std::runtime_error("FOCAS_DLL_OR_CONNECTION_EXPORT_MISSING");
        std::string host = env("FOCAS_HOST", "127.0.0.1");
        int port = std::stoi(env("FOCAS_PORT", "8193"));
        if (port < 1 || port > 65535) throw std::runtime_error("Invalid FOCAS_PORT");
        short ret = ctx.call("CONNECT", "cnc_allclibhndl3", "host=" + host + ";port=" + std::to_string(port) + ";timeout=10",
            api.cnc_allclibhndl3, [&] { return api.cnc_allclibhndl3(host.c_str(), static_cast<unsigned short>(port), 10, &handle); },
            [&](short) { return "handle=" + std::to_string(handle); });
        if (ret != EW_OK) result = 3;
        else {
            connected = true;
            bool alarm_snapshot_captured = false;
            ctx.set_error_diagnostics([&](bool include_error_detail) {
                if (include_error_detail && api.cnc_getdtailerr) {
                    ODBERR detail{};
                    ctx.call("ERROR_DIAGNOSTIC", "cnc_getdtailerr",
                        "FlibHndl=" + std::to_string(handle), api.cnc_getdtailerr,
                        [&] { return api.cnc_getdtailerr(handle, &detail); }, [&](short) {
                            return "err_no=" + std::to_string(detail.err_no) +
                                ";err_dtno=" + std::to_string(detail.err_dtno);
                        });
                } else if (include_error_detail) {
                    ctx.host("cnc_getdtailerr", -15, "UNSUPPORTED_SYMBOL", "detail error reader is not exported");
                }
                if (!alarm_snapshot_captured) {
                    alarm_snapshot_captured = capture_alarm_diagnostics(api, handle, ctx);
                }
            });
            ODBST initial_state{};
            short initial_rc = ctx.call("ALARM_PREFLIGHT", "cnc_statinfo",
                "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
                [&] { return api.cnc_statinfo(handle, &initial_state); }, [&](short) {
                    return StatinfoData(initial_state);
                });
            bool recovery_ready = initial_state.alarm == 0 && initial_state.emergency == 0;
            if (initial_rc == EW_OK && !recovery_ready) recovery_ready = recover_alarm(api, handle, ctx);
            if (initial_rc != EW_OK || (!recovery_ready && !__ALLOW_ALARM_READS__)) {
                ctx.block("scenario", "alarm_recovery_unverified");
                ret = 12;
            } else if (__REQUIRE_REMOTE__ && initial_state.aut != 10 &&
                       !ctx.wait_for_operator(api, handle, "remote", "selected_DNC_APIs")) {
                ctx.block("selected_DNC_APIs", "remote_mode_not_verified_after_operator_timeout");
                ret = 12;
            } else {
                if (!recovery_ready) {
                    ctx.host("alarm_recovery", 0, "ALARM_PRESENT_READ_ONLY_CONTINUE",
                        "manual_action_timed_out=true;state_changing_targets_excluded=true");
                }
                ret = __EXECUTION_CALL__;
                ODBST final_state{};
                if (api.cnc_statinfo) {
                    short final_rc = ctx.call("ALARM_POSTCHECK", "cnc_statinfo",
                        "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
                        [&] { return api.cnc_statinfo(handle, &final_state); }, [&](short) {
                            return StatinfoData(final_state);
                        });
                    if (final_rc == EW_OK && final_state.alarm && !ctx.manual_action_timed_out()) recover_alarm(api, handle, ctx);
                }
            }
            if (ret != EW_OK) ctx.diagnose(true);
            if (ret != EW_OK || ctx.failed()) result = 4;
        }
    } catch (const std::exception& exc) {
        ctx.host("host_runtime_exception", -1, "RUNTIME_EXCEPTION", exc.what());
        result = 5;
    }
    if (connected) {
        short ret = ctx.call("DISCONNECT", "cnc_freelibhndl", "FlibHndl=" + std::to_string(handle),
            api.cnc_freelibhndl, [&] { return api.cnc_freelibhndl(handle); }, [](short) { return ""; });
        if (ret != EW_OK) result = 6;
    }
    Sleep(30);
    sniffer.Stop();
    if (sniffer.Failed()) { ctx.host("host_capture_error", -1, "PCAP_CAPTURE_FAILED", "Npcap dispatch or flush failed"); result = 2; }
    return result;
}
