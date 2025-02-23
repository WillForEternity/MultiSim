// quadruped.h

#ifndef QUADRUPED_H
#define QUADRUPED_H

#include <ode/ode.h>
#include "common.h"  

#ifndef NUM_RAYS
#define SENSOR_RAY_GRID   5
#define NUM_RAYS          (SENSOR_RAY_GRID * SENSOR_RAY_GRID)
#endif

typedef struct Quadruped {
    dBodyID body;
    dGeomID bodyGeom;
    dBodyID leg[4];
    dGeomID legGeom[4];
    dJointID hip[4];
    dBodyID wheel[4];
    dGeomID wheelTransform[4];
    dJointID wheelJoint[4];
    double fitness;
    
    // Movement and reward tracking.
    double spawnX;
    double spawnTime;
    double prevX;
    double prevY;
    double lastMoveTime;
    bool stagnancyPunished;
    
    // Reshuffle tracking.
    int respawnCount;
    double lastRespawnTime;
    
    // Collision penalty accumulator.
    double collisionPenalty;
    
    // Sensor rays.
    dGeomID raySensors[NUM_RAYS];
    double sensorValues[NUM_RAYS];
    
    // Distance measure.
    double distanceToTarget;
    double prevTargetDistance;
    
    // For TD bootstrapping (critic target).
    double prevCriticValue;
    
    // For visual blinking feedback.
    int blinkType;
    int blinkCount;
    double blinkStartTime;
} Quadruped;

#endif // QUADRUPED_H
