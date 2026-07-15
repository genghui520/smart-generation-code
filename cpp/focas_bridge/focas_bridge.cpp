#include <windows.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

struct ODBST {
    short hdck;
    short tmmode;
    short aut;
    short run;
    short motion;
    short mstb;
    short emergency;
    short alarm;
    short edit;
};

struct ODBACT {
    short dummy[2];
    long data;
};

struct POSELM {
    long data;
    short dec;
    short unit;
    short disp;
    char name;
    char suff;
};

struct ODBPOS {
    POSELM abs;
    POSELM mach;
    POSELM rel;
    POSELM dist;
};

using cnc_allclibhndl3_t = short(__stdcall *)(const char *, unsigned short, long, unsigned short *);
using cnc_freelibhndl_t = short(__stdcall *)(unsigned short);
using cnc_statinfo_t = short(__stdcall *)(unsigned short, ODBST *);
using cnc_actf_t = short(__stdcall *)(unsigned short, ODBACT *);
using cnc_acts_t = short(__stdcall *)(unsigned short, ODBACT *);
using cnc_rdposition_t = short(__stdcall *)(unsigned short, short, short *, ODBPOS *);
using cnc_alarm2_t = short(__stdcall *)(unsigned short, long *);

struct Config {
    std::wstring install_dir = L"D:\\Program Files (x86)\\FANUC\\NCGuide FS0i-F";
    std::string host = "127.0.0.1";
    unsigned short port = 8194;
    long timeout_seconds = 10;
    std::string action = "read_run_status";
};

std::string json_escape(const std::string &value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    return out.str();
}

std::string narrow(const std::wstring &value) {
    if (value.empty()) {
        return "";
    }
    int size = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        return "";
    }
    std::string result(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.c_str(), -1, result.data(), size, nullptr, nullptr);
    return result;
}

void print_error(int status_code, const std::string &error) {
    std::cout << "{\"status_code\":" << status_code
              << ",\"error\":\"" << json_escape(error) << "\"}" << std::endl;
}

std::wstring arg_w(int argc, wchar_t **argv, int &index) {
    if (index + 1 >= argc) {
        return L"";
    }
    index += 1;
    return argv[index];
}

std::string to_utf8(const std::wstring &value) {
    return narrow(value);
}

Config parse_args(int argc, wchar_t **argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::wstring key = argv[i];
        if (key == L"--install-dir") {
            config.install_dir = arg_w(argc, argv, i);
        } else if (key == L"--host") {
            config.host = to_utf8(arg_w(argc, argv, i));
        } else if (key == L"--port") {
            config.port = static_cast<unsigned short>(std::stoi(arg_w(argc, argv, i)));
        } else if (key == L"--timeout") {
            config.timeout_seconds = std::stol(arg_w(argc, argv, i));
        } else if (key == L"--action") {
            config.action = to_utf8(arg_w(argc, argv, i));
        }
    }
    return config;
}

class FocasBridge {
public:
    explicit FocasBridge(Config config) : config_(std::move(config)) {}

    ~FocasBridge() {
        if (handle_ != 0 && cnc_freelibhndl_ != nullptr) {
            cnc_freelibhndl_(handle_);
        }
        if (dll_ != nullptr) {
            FreeLibrary(dll_);
        }
    }

    bool load_library() {
        SetDllDirectoryW(config_.install_dir.c_str());
        std::wstring dll_path = config_.install_dir + L"\\Fwlib32.dll";
        dll_ = LoadLibraryW(dll_path.c_str());
        if (dll_ == nullptr) {
            print_error(500, "LoadLibraryW failed for " + narrow(dll_path));
            return false;
        }
        cnc_allclibhndl3_ = reinterpret_cast<cnc_allclibhndl3_t>(GetProcAddress(dll_, "cnc_allclibhndl3"));
        cnc_freelibhndl_ = reinterpret_cast<cnc_freelibhndl_t>(GetProcAddress(dll_, "cnc_freelibhndl"));
        cnc_statinfo_ = reinterpret_cast<cnc_statinfo_t>(GetProcAddress(dll_, "cnc_statinfo"));
        cnc_actf_ = reinterpret_cast<cnc_actf_t>(GetProcAddress(dll_, "cnc_actf"));
        cnc_acts_ = reinterpret_cast<cnc_acts_t>(GetProcAddress(dll_, "cnc_acts"));
        cnc_rdposition_ = reinterpret_cast<cnc_rdposition_t>(GetProcAddress(dll_, "cnc_rdposition"));
        cnc_alarm2_ = reinterpret_cast<cnc_alarm2_t>(GetProcAddress(dll_, "cnc_alarm2"));
        if (
            cnc_allclibhndl3_ == nullptr ||
            cnc_freelibhndl_ == nullptr ||
            cnc_statinfo_ == nullptr ||
            cnc_actf_ == nullptr ||
            cnc_acts_ == nullptr ||
            cnc_rdposition_ == nullptr ||
            cnc_alarm2_ == nullptr
        ) {
            print_error(500, "Required FOCAS functions were not exported by Fwlib32.dll");
            return false;
        }
        return true;
    }

    short connect() {
        if (dll_ == nullptr && !load_library()) {
            return 500;
        }
        unsigned short handle = 0;
        short result = cnc_allclibhndl3_(
            config_.host.c_str(),
            config_.port,
            config_.timeout_seconds,
            &handle
        );
        if (result == 0) {
            handle_ = handle;
        }
        return result;
    }

    int run() {
        if (config_.action == "probe") {
            if (!load_library()) {
                return 1;
            }
            std::cout << "{\"status_code\":0,\"loaded\":true,\"library\":\"Fwlib32.dll\",\"bridge\":\"cpp\"}" << std::endl;
            return 0;
        }
        if (config_.action == "connect") {
            short result = connect();
            if (result != 0) {
                std::cout << "{\"status_code\":" << result
                          << ",\"error\":\"cnc_allclibhndl3 failed\""
                          << ",\"host\":\"" << json_escape(config_.host) << "\""
                          << ",\"port\":" << config_.port
                          << ",\"bridge\":\"cpp\"}" << std::endl;
                return 1;
            }
            std::cout << "{\"status_code\":0,\"handle\":" << handle_
                      << ",\"host\":\"" << json_escape(config_.host) << "\""
                      << ",\"port\":" << config_.port
                      << ",\"bridge\":\"cpp\"}" << std::endl;
            return 0;
        }
        if (config_.action == "read_run_status") {
            short result = connect();
            if (result != 0) {
                print_connect_error(result);
                return 1;
            }
            ODBST status{};
            short stat_result = cnc_statinfo_(handle_, &status);
            std::cout << "{\"status_code\":" << stat_result
                      << ",\"function\":\"cnc_statinfo\""
                      << ",\"bridge\":\"cpp\""
                      << ",\"statinfo\":{"
                      << "\"hdck\":" << status.hdck
                      << ",\"tmmode\":" << status.tmmode
                      << ",\"aut\":" << status.aut
                      << ",\"run\":" << status.run
                      << ",\"motion\":" << status.motion
                      << ",\"mstb\":" << status.mstb
                      << ",\"emergency\":" << status.emergency
                      << ",\"alarm\":" << status.alarm
                      << ",\"edit\":" << status.edit
                      << "}}" << std::endl;
            return stat_result == 0 ? 0 : 1;
        }
        if (config_.action == "read_feed_speed") {
            short result = connect();
            if (result != 0) {
                print_connect_error(result);
                return 1;
            }
            ODBACT feed{};
            short feed_result = cnc_actf_(handle_, &feed);
            std::cout << "{\"status_code\":" << feed_result
                      << ",\"function\":\"cnc_actf\""
                      << ",\"bridge\":\"cpp\""
                      << ",\"feed_speed\":" << feed.data
                      << "}" << std::endl;
            return feed_result == 0 ? 0 : 1;
        }
        if (config_.action == "read_spindle_speed") {
            short result = connect();
            if (result != 0) {
                print_connect_error(result);
                return 1;
            }
            ODBACT spindle{};
            short spindle_result = cnc_acts_(handle_, &spindle);
            std::cout << "{\"status_code\":" << spindle_result
                      << ",\"function\":\"cnc_acts\""
                      << ",\"bridge\":\"cpp\""
                      << ",\"spindle_speed\":" << spindle.data
                      << "}" << std::endl;
            return spindle_result == 0 ? 0 : 1;
        }
        if (config_.action == "read_alarm") {
            short result = connect();
            if (result != 0) {
                print_connect_error(result);
                return 1;
            }
            long alarm_bits = 0;
            short alarm_result = cnc_alarm2_(handle_, &alarm_bits);
            std::cout << "{\"status_code\":" << alarm_result
                      << ",\"function\":\"cnc_alarm2\""
                      << ",\"bridge\":\"cpp\""
                      << ",\"alarm_bits\":" << alarm_bits
                      << "}" << std::endl;
            return alarm_result == 0 ? 0 : 1;
        }
        if (config_.action == "read_position") {
            short result = connect();
            if (result != 0) {
                print_connect_error(result);
                return 1;
            }
            short type = 1;
            short num = 3;
            ODBPOS positions[3]{};
            short position_result = cnc_rdposition_(handle_, type, &num, positions);
            std::cout << "{\"status_code\":" << position_result
                      << ",\"function\":\"cnc_rdposition\""
                      << ",\"bridge\":\"cpp\""
                      << ",\"position_type\":\"machine\""
                      << ",\"axis_count\":" << num
                      << ",\"positions\":[";
            for (int i = 0; i < num; ++i) {
                if (i > 0) {
                    std::cout << ",";
                }
                std::cout << "{\"axis\":\"" << positions[i].mach.name
                          << "\",\"data\":" << positions[i].mach.data
                          << ",\"dec\":" << positions[i].mach.dec
                          << ",\"unit\":" << positions[i].mach.unit
                          << "}";
            }
            std::cout << "]}" << std::endl;
            return position_result == 0 ? 0 : 1;
        }
        print_error(400, "Unsupported action: " + config_.action);
        return 1;
    }

private:
    void print_connect_error(short result) {
        std::cout << "{\"status_code\":" << result
                  << ",\"error\":\"cnc_allclibhndl3 failed\""
                  << ",\"host\":\"" << json_escape(config_.host) << "\""
                  << ",\"port\":" << config_.port
                  << ",\"bridge\":\"cpp\"}" << std::endl;
    }

    Config config_;
    HMODULE dll_ = nullptr;
    unsigned short handle_ = 0;
    cnc_allclibhndl3_t cnc_allclibhndl3_ = nullptr;
    cnc_freelibhndl_t cnc_freelibhndl_ = nullptr;
    cnc_statinfo_t cnc_statinfo_ = nullptr;
    cnc_actf_t cnc_actf_ = nullptr;
    cnc_acts_t cnc_acts_ = nullptr;
    cnc_rdposition_t cnc_rdposition_ = nullptr;
    cnc_alarm2_t cnc_alarm2_ = nullptr;
};

int wmain(int argc, wchar_t **argv) {
    Config config = parse_args(argc, argv);
    FocasBridge bridge(config);
    return bridge.run();
}
