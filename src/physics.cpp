// physics.cpp
// ODE physics utilities for body/joint creation and world management.

#include "physics.h"
#include "config.h"
#include <cmath>

namespace physics {

void initWorld(dWorldID& world, dSpaceID& space, dJointGroupID& contactGroup, dGeomID& groundPlane) {
    dInitODE();
    world = dWorldCreate();
    space = dHashSpaceCreate(0);
    contactGroup = dJointGroupCreate(0);
    
    dWorldSetGravity(world, 0, 0, -9.81);
    dWorldSetCFM(world, 1e-5);
    dWorldSetERP(world, 0.2);
    dWorldSetQuickStepNumIterations(world, 50);
    
    groundPlane = dCreatePlane(space, 0, 0, 1, 0);
}

void cleanup(dWorldID world, dSpaceID space, dJointGroupID contactGroup) {
    dJointGroupDestroy(contactGroup);
    dSpaceDestroy(space);
    dWorldDestroy(world);
    dCloseODE();
}

// Wall type enum for collision handling
enum WallType { SPAWN_WALL = 0, SIDE_WALL = 1, FAR_WALL = 2 };

void createBoundaryWalls(dSpaceID space, dGeomID boundaryBoxes[4]) {
    using namespace config;
    
    const double wallHeight = 5.0;
    const double wallThickness = 1.0;
    const double xRange = WORLD_X_MAX - WORLD_X_MIN;
    const double yRange = WORLD_Y_MAX - WORLD_Y_MIN;
    
    // Top wall (positive Y)
    boundaryBoxes[0] = dCreateBox(space, xRange + wallThickness, wallThickness, wallHeight);
    dGeomSetPosition(boundaryBoxes[0], (WORLD_X_MAX + WORLD_X_MIN) / 2.0, WORLD_Y_MAX + 0.5, wallHeight / 2.0);
    dGeomSetData(boundaryBoxes[0], reinterpret_cast<void*>(static_cast<intptr_t>(SIDE_WALL)));
    
    // Bottom wall (negative Y)
    boundaryBoxes[1] = dCreateBox(space, xRange + wallThickness, wallThickness, wallHeight);
    dGeomSetPosition(boundaryBoxes[1], (WORLD_X_MAX + WORLD_X_MIN) / 2.0, WORLD_Y_MIN - 0.5, wallHeight / 2.0);
    dGeomSetData(boundaryBoxes[1], reinterpret_cast<void*>(static_cast<intptr_t>(SIDE_WALL)));
    
    // Left wall (spawn wall, negative X)
    boundaryBoxes[2] = dCreateBox(space, wallThickness, yRange + wallThickness, wallHeight);
    dGeomSetPosition(boundaryBoxes[2], WORLD_X_MIN - 0.5, 0.0, wallHeight / 2.0);
    dGeomSetData(boundaryBoxes[2], reinterpret_cast<void*>(static_cast<intptr_t>(SPAWN_WALL)));
    
    // Right wall (far wall, positive X)
    boundaryBoxes[3] = dCreateBox(space, wallThickness, yRange + wallThickness, wallHeight);
    dGeomSetPosition(boundaryBoxes[3], WORLD_X_MAX + 0.5, 0.0, wallHeight / 2.0);
    dGeomSetData(boundaryBoxes[3], reinterpret_cast<void*>(static_cast<intptr_t>(FAR_WALL)));
}

void createTargetBall(dWorldID world, dSpaceID space, dBodyID& targetBall, dGeomID& targetBallGeom) {
    using namespace config;
    
    targetBall = dBodyCreate(world);
    dMass m;
    dMassSetSphere(&m, 2.0, TARGET_RADIUS);
    dBodySetMass(targetBall, &m);
    
    targetBallGeom = dCreateSphere(space, TARGET_RADIUS);
    dGeomSetBody(targetBallGeom, targetBall);
    
    // Place target on ground (z = radius so it sits on ground)
    dBodySetPosition(targetBall, TARGET_X, TARGET_Y, TARGET_RADIUS);
}

dBodyID createBox(dWorldID world, dSpaceID space,
                  double length, double width, double height,
                  double mass, dGeomID* geomOut) {
    dMass m;
    dBodyID body = dBodyCreate(world);
    dMassSetBoxTotal(&m, mass, length, width, height);
    dBodySetMass(body, &m);
    
    dGeomID geom = dCreateBox(space, length, width, height);
    dGeomSetBody(geom, body);
    
    if (geomOut) {
        *geomOut = geom;
    }
    return body;
}

dBodyID createCylinder(dWorldID world, dSpaceID space,
                       double radius, double length, double mass,
                       bool isWheel, dGeomID* wheelTransformOut) {
    dMass m;
    dBodyID body = dBodyCreate(world);
    dMassSetCylinderTotal(&m, mass, 3, radius, length);
    dBodySetMass(body, &m);
    
    if (isWheel) {
        // Create transform geometry with rotated cylinder for wheel
        dGeomID cylinderGeom = dCreateCylinder(nullptr, radius, length);
        dGeomID transGeom = dCreateGeomTransform(space);
        dGeomTransformSetGeom(transGeom, cylinderGeom);
        dGeomSetBody(transGeom, body);
        
        dMatrix3 R;
        dRFromAxisAndAngle(R, 1, 0, 0, M_PI / 2);
        dGeomSetRotation(transGeom, R);
        
        if (wheelTransformOut) {
            *wheelTransformOut = transGeom;
        }
    } else {
        dGeomID geom = dCreateCylinder(space, radius, length);
        dGeomSetBody(geom, body);
    }
    
    return body;
}

void createQuadruped(dWorldID world, dSpaceID space,
                     Quadruped* quad, double x, double y,
                     double simTime, dBodyID targetBall,
                     dGeomID* obstacles, dVector3* obstacleSizes, int numObstacles) {
    using namespace config;
    
    // Calculate anchor positions for the 4 legs
    double anchorX[4], anchorY[4];
    for (int i = 0; i < 4; i++) {
        double signX = (i < 2) ? -1.0 : 1.0;
        double signY = ((i % 2) == 0) ? -1.0 : 1.0;
        anchorX[i] = x + signX * (BODY_LENGTH * 0.5);
        anchorY[i] = y + signY * (BODY_WIDTH * 0.5);
    }
    
    // Calculate base height (check for obstacles underneath)
    double baseHeight = 0.0;
    for (int i = 0; i < numObstacles; i++) {
        const dReal* pos = dGeomGetPosition(obstacles[i]);
        double sx = obstacleSizes[i][0];
        double sy = obstacleSizes[i][1];
        double half_sx = sx / 2.0;
        double half_sy = sy / 2.0;
        
        if (x >= pos[0] - half_sx && x <= pos[0] + half_sx &&
            y >= pos[1] - half_sy && y <= pos[1] + half_sy) {
            double top = pos[2] + (obstacleSizes[i][2] / 2.0);
            if (top > baseHeight) {
                baseHeight = top;
            }
        }
    }
    
    double hipZ = baseHeight + LEG_LENGTH + 0.1;
    double torsoZ = hipZ + (BODY_HEIGHT * 0.5);
    
    // Create main body
    quad->body = createBox(world, space, BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT, BODY_MASS, &quad->bodyGeom);
    dBodySetPosition(quad->body, x, y, torsoZ);
    dBodySetData(quad->body, quad);
    quad->fitness = x;
    
    // Initialize tracking state
    quad->spawnX = x;
    quad->spawnTime = simTime;
    quad->prevX = x;
    quad->prevY = y;
    quad->lastMoveTime = simTime;
    quad->stagnancyPunished = false;
    quad->respawnCount = 0;
    quad->lastRespawnTime = simTime;
    quad->blinkType = BLINK_NONE;
    quad->blinkCount = 0;
    quad->blinkStartTime = 0.0;
    quad->collisionPenalty = 0.0;
    quad->ballVisible = false;
    quad->ballVisionValue = 1.0;
    quad->ballRayIndex = -1;
    
    // Calculate initial distance to target
    const dReal* tPos = dBodyGetPosition(targetBall);
    double dx = tPos[0] - x;
    double dy = tPos[1] - y;
    double dz = tPos[2] - torsoZ;
    double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    quad->distanceToTarget = dist;
    quad->prevTargetDistance = dist;
    quad->prevCriticValue = 0.0;
    
    // Store target position
    quad->targetX = tPos[0];
    quad->targetY = tPos[1];
    
    // Initialize position and orientation
    quad->x = x;
    quad->y = y;
    quad->orientation = 0.0;
    
    // Create sensor rays
    for (int i = 0; i < NUM_RAYS; i++) {
        quad->raySensors[i] = dCreateRay(space, SENSOR_MAX_DIST);
        dGeomRaySet(quad->raySensors[i], x, y, torsoZ, 1, 0, 0);
        quad->sensorValues[i] = 1.0;
    }
    
    // Create legs and wheels
    for (int i = 0; i < 4; i++) {
        // Create leg
        quad->leg[i] = createBox(world, space, LEG_WIDTH, LEG_WIDTH, LEG_LENGTH, LEG_MASS, &quad->legGeom[i]);
        dBodySetPosition(quad->leg[i], anchorX[i], anchorY[i], hipZ - LEG_LENGTH * 0.5);
        dBodySetData(quad->leg[i], quad);
        
        // Create hip joint
        quad->hip[i] = dJointCreateHinge(world, nullptr);
        dJointAttach(quad->hip[i], quad->body, quad->leg[i]);
        dJointSetHingeAnchor(quad->hip[i], anchorX[i], anchorY[i], hipZ);
        dJointSetHingeAxis(quad->hip[i], 0, 1, 0);
        dJointSetHingeParam(quad->hip[i], dParamFMax, HIP_FMAX);
        dJointSetHingeParam(quad->hip[i], dParamERP, 0.8);
        dJointSetHingeParam(quad->hip[i], dParamCFM, 1e-5);
        
        // Lock hips at fixed angles
        if (i < 2) {
            dJointSetHingeParam(quad->hip[i], dParamLoStop, -1.0);
            dJointSetHingeParam(quad->hip[i], dParamHiStop, -1.0);
        } else {
            dJointSetHingeParam(quad->hip[i], dParamLoStop, 1.0);
            dJointSetHingeParam(quad->hip[i], dParamHiStop, 1.0);
        }
        
        // Create wheel
        quad->wheel[i] = createCylinder(world, space, WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS, true, &quad->wheelTransform[i]);
        double offsetY = ((i % 2) == 0) ? -(LEG_WIDTH / 2.0 + WHEEL_RADIUS) : (LEG_WIDTH / 2.0 + WHEEL_RADIUS);
        dBodySetPosition(quad->wheel[i], anchorX[i], anchorY[i] + offsetY, hipZ - LEG_LENGTH);
        dBodySetData(quad->wheel[i], quad);
        
        // Create wheel joint
        quad->wheelJoint[i] = dJointCreateHinge(world, nullptr);
        dJointAttach(quad->wheelJoint[i], quad->leg[i], quad->wheel[i]);
        dJointSetHingeAnchor(quad->wheelJoint[i], anchorX[i], anchorY[i] + offsetY, hipZ - LEG_LENGTH);
        dJointSetHingeAxis(quad->wheelJoint[i], 0, 1, 0);
        dJointSetHingeParam(quad->wheelJoint[i], dParamLoStop, -dInfinity);
        dJointSetHingeParam(quad->wheelJoint[i], dParamHiStop, dInfinity);
        dJointSetHingeParam(quad->wheelJoint[i], dParamFMax, WHEEL_FMAX);
        dJointSetHingeParam(quad->wheelJoint[i], dParamERP, 80);
        dJointSetHingeParam(quad->wheelJoint[i], dParamCFM, 1e-4);
    }
}

void destroyQuadruped(Quadruped* quad) {
    if (quad->body) {
        dBodyDestroy(quad->body);
        quad->body = nullptr;
    }
    if (quad->bodyGeom) {
        dGeomDestroy(quad->bodyGeom);
        quad->bodyGeom = nullptr;
    }
    
    for (int j = 0; j < 4; j++) {
        if (quad->leg[j]) {
            dBodyDestroy(quad->leg[j]);
            quad->leg[j] = nullptr;
        }
        if (quad->legGeom[j]) {
            dGeomDestroy(quad->legGeom[j]);
            quad->legGeom[j] = nullptr;
        }
        if (quad->hip[j]) {
            dJointDestroy(quad->hip[j]);
            quad->hip[j] = nullptr;
        }
        if (quad->wheelJoint[j]) {
            dJointDestroy(quad->wheelJoint[j]);
            quad->wheelJoint[j] = nullptr;
        }
        if (quad->wheel[j]) {
            dBodyDestroy(quad->wheel[j]);
            quad->wheel[j] = nullptr;
        }
        if (quad->wheelTransform[j]) {
            dGeomDestroy(quad->wheelTransform[j]);
            quad->wheelTransform[j] = nullptr;
        }
    }
    
    for (int k = 0; k < config::NUM_RAYS; k++) {
        if (quad->raySensors[k]) {
            dGeomDestroy(quad->raySensors[k]);
            quad->raySensors[k] = nullptr;
        }
    }
}

bool isBadRotation(const dMatrix3 R) {
    double norm = std::sqrt(R[0] * R[0] + R[4] * R[4] + R[8] * R[8]);
    return (norm < 1e-3 || norm > 1e3);
}

void safeNormalizeBodyRotation(dBodyID body) {
    const dReal* R = dBodyGetRotation(body);
    if (isBadRotation(R)) {
        dMatrix3 identity;
        dRSetIdentity(identity);
        dBodySetRotation(body, identity);
    }
}

} // namespace physics
