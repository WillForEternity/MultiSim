#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

// Include C++ headers outside of extern "C"
#include "quadruped.h"  
#include "common.h"

// Only wrap the API declarations with C linkage if needed.
#ifdef __cplusplus
extern "C" {
#endif

void initODE(void);
void cleanupODE(void);
void createStaticObstacles(int totalObstacles, double innerMargin, double outerMargin);

#ifdef __cplusplus
}
#endif

#endif // ENVIRONMENT_H
