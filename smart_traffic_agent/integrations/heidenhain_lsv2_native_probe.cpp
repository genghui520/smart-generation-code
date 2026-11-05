#include <windows.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "lsv2/lsv2_all.h"

static double arg_or(const char* value, double fallback) {
    return value ? std::atof(value) : fallback;
}

int main(int argc, char** argv) {
    const char* host = argc > 1 ? argv[1] : "192.168.56.102";
    const double coord[3] = {
        arg_or(argc > 2 ? argv[2] : nullptr, 0.0),
        arg_or(argc > 3 ? argv[3] : nullptr, 0.0),
        arg_or(argc > 4 ? argv[4] : nullptr, 0.0),
    };
    const double orientation[3] = {
        arg_or(argc > 5 ? argv[5] : nullptr, 0.0),
        arg_or(argc > 6 ? argv[6] : nullptr, 0.0),
        arg_or(argc > 7 ? argv[7] : nullptr, 0.0),
    };

    HANDLE port = nullptr;
    if (!LSV2OpenTCP(&port, host)) {
        std::cout << "{\"success\":false,\"stage\":\"open\",\"win_error\":"
                  << static_cast<unsigned long>(GetLastError()) << "}\n";
        return 2;
    }

    bool logged_in = LSV2Login(port, PW_STREAMAXES, nullptr) != FALSE;
    if (!logged_in) {
        std::cout << "{\"success\":false,\"stage\":\"login\",\"keyword\":\"STREAMAXES\",\"win_error\":"
                  << static_cast<unsigned long>(GetLastError()) << "}\n";
        LSV2Close(port);
        return 3;
    }

    tMachinePositions positions{};
    const BOOL ok = LSV2CalcMachinePositions(port, coord, orientation, &positions);
    std::cout << std::setprecision(17)
              << "{\"success\":" << (ok ? "true" : "false")
              << ",\"api\":\"LSV2CalcMachinePositions\""
              << ",\"coord\":[" << coord[0] << "," << coord[1] << "," << coord[2] << "]"
              << ",\"orientation\":[" << orientation[0] << "," << orientation[1] << "," << orientation[2] << "]"
              << ",\"count\":" << positions.Count
              << ",\"valid_axes_mask\":" << positions.ValidAxesMask
              << ",\"axes_values_1\":[";
    for (int i = 0; i < LSV2_MAX_AXES; ++i) {
        if (i) std::cout << ",";
        std::cout << positions.AxesValues1[i];
    }
    std::cout << "] ,\"axes_values_2\":[";
    for (int i = 0; i < LSV2_MAX_AXES; ++i) {
        if (i) std::cout << ",";
        std::cout << positions.AxesValues2[i];
    }
    std::cout << "]";
    if (!ok) {
        std::cout << ",\"win_error\":" << static_cast<unsigned long>(GetLastError());
    }
    std::cout << "}\n";

    LSV2Logout(port, PW_STREAMAXES);
    LSV2Close(port);
    return ok ? 0 : 4;
}
