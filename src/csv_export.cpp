// csv_export.cpp
// File writing utilities for network parameter visualization.

#include "csv_export.h"
#include <cstdio>

void writeCSVData(const char* filename, const char* data) {
    FILE* fp = std::fopen(filename, "w");
    if (!fp) {
        std::perror("Error opening CSV file");
        return;
    }
    std::fprintf(fp, "%s", data);
    std::fclose(fp);
}
