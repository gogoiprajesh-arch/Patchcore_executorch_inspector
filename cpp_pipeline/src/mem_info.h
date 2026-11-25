#pragma once
#include <vector>
#include <string>    
#include <algorithm>  
#include <numeric>
#include <fstream>
#include <sstream>

struct MemInfo {
    long rss_kb = 0, 
    peak_kb = 0, 
    anon_kb = 0, 
    file_kb = 0;
};

MemInfo read_mem();

void print_mem(const char* tag);

