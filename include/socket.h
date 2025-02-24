#ifndef SOCKET_H
#define SOCKET_H

#ifdef __cplusplus
extern "C" {
#endif

// Writes the given CSV-formatted data to a file specified by filename.
void writeCSVData(const char* filename, const char* data);

#ifdef __cplusplus
}
#endif

#endif // SOCKET_H
