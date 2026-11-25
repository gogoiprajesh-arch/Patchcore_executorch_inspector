#include "mem_info.h"
#include <iostream>

MemInfo read_mem() {
    MemInfo m;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key;
        long val = 0;
        ss >> key >> val;
        if      (key == "VmRSS:")  m.rss_kb  = val;
        else if (key == "VmHWM:")  m.peak_kb = val;
        else if (key == "RssAnon:") m.anon_kb = val;
        else if (key == "RssFile:") m.file_kb = val;
    }
    return m;
}

void print_mem(const char* tag) {
    MemInfo m = read_mem();
    std::cout << "[mem] " << tag << " | RSS " << m.rss_kb / 1024.0
              << " MB | anon " << m.anon_kb / 1024.0
              << " MB | file " << m.file_kb / 1024.0
              << " MB | peak " << m.peak_kb / 1024.0 << " MB\n";
}