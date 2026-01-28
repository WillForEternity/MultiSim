#pragma once

// === CSV Export ===
// File writing utilities for network parameter visualization.

/// Write CSV data to a file.
/// @param filename Path to output file
/// @param data CSV-formatted string to write
void writeCSVData(const char* filename, const char* data);
