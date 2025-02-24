#include "../include/socket.h"
#include <cstdio>

void writeCSVData(const char* filename, const char* data) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening CSV file");
        return;
    }
    fprintf(fp, "%s", data);
    fclose(fp);
}
