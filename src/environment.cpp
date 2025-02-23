// environment.cpp

// compile: g++-14 -stdlib=libc++ -I/usr/local/include -L/usr/local/lib -o multiSim2 src/environment.cpp src/neural_network.cpp -lode -ldrawstuff -lm -framework GLUT -framework OpenGL -fopenmp

#define GL_SILENCE_DEPRECATION

#include "../include/environment.h"
#include "../include/neural_network.h"
#include "../include/common.h"
#include "../include/quadruped.h"
#include <ode/ode.h>
#include <drawstuff/drawstuff.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <GLUT/glut.h>

// --- Visual behavior feedback ---
#define BLINK_NONE  0
#define BLINK_BLUE  1   // for considerable weight change
#define BLINK_GREEN 2   // for reward
#define BLINK_RED   3   // for punishment

#define ACTOR_OUTPUTS 4

// External neural network functions.
extern "C" {
    // Defined in neural_network.cpp.
    // void initNetwork();
    // void runNeuralNetwork(struct Quadruped* quad, double reward, double out_actions[ACTOR_OUTPUTS]);
    extern double globalWeightChangeBlink;
    extern bool globalInitNetworkCalled;
}

// Simulation and geometry settings.
#define NUM_QUADRUPEDS    50
#define SIMULATION_DT     0.015

// Sensor parameters.
#define SENSOR_RAY_GRID   5
#define NUM_RAYS          (SENSOR_RAY_GRID * SENSOR_RAY_GRID)
#define SENSOR_MAX_DIST   4.0
#define SENSOR_CONE_ANGLE (M_PI/6.0)
#define DRAWN_MAX_LENGTH  4.0

// Body and leg parameters.
#define BODY_LENGTH       0.4
#define BODY_WIDTH        0.15
#define BODY_HEIGHT       0.05
#define BODY_MASS         10.0

// Leg parameters.
#define LEG_WIDTH         0.05
#define LEG_LENGTH        0.3
#define LEG_MASS          8.0

// Wheel parameters.
#define WHEEL_RADIUS      0.1
#define WHEEL_WIDTH       0.1
#define WHEEL_MASS        8.0

#define HIP_FMAX          1000.0
#define WHEEL_FMAX        1000.0
#define MAX_OBSTACLES     10000

// Behavior parameters.
#define STAGNATION_TIME    20000.0
#define SIDE_WALL_THRESHOLD 0.3

// Reshuffle condition.
#define RESHUFFLE_RESPAWN_THRESHOLD 4  
#define RESHUFFLE_TIME_WINDOW       60.0  

// Blink parameters.
#define BLINK_PERIOD 0.2

/******************************************************************************
 * Data Structures
 ******************************************************************************/
typedef struct {
    Quadruped quads[NUM_QUADRUPEDS];
} WorldObjects;

static WorldObjects wobjects;

/******************************************************************************
 * ODE/Drawstuff Globals
 ******************************************************************************/
static dWorldID      world;
static dSpaceID      space;
static dJointGroupID contactGroup;
static dGeomID       groundPlaneGeom;
static dGeomID       boundaryBoxes[4];

enum WallType { SPAWN_WALL, SIDE_WALL, FAR_WALL };

static double simulationTime = 0.0;
static dGeomID obstacles[MAX_OBSTACLES];
static dVector3 obstacleSizes[MAX_OBSTACLES];
static int totalObstaclesCreated = 0;

static dBodyID targetBall;
static dGeomID targetBallGeom;

/******************************************************************************
 * Forward declarations.
 ******************************************************************************/
static void simLoop(int pause);
static void start(void);
static void command(int cmd);
static void stop(void);
static double getEnvironmentHeight(double x, double y);
static void replaceQuadruped(int index);
static double randomUniform(double min, double max);
static void updateSensorsAndControl(Quadruped *quad);
static void createQuadruped(Quadruped *quad, double x, double y);
static void safeNormalizeBodyRotation(dBodyID body);
static void drawText(const char* text, float x, float y);
static void drawObstacleLines(void);

/******************************************************************************
 * Helper: Check if rotation matrix is invalid.
 ******************************************************************************/
static bool isBadRotation(const dMatrix3 R) {
    double norm = sqrt(R[0]*R[0] + R[4]*R[4] + R[8]*R[8]);
    return (norm < 1e-3 || norm > 1e3);
}

/******************************************************************************
 * safeNormalizeBodyRotation: Replace invalid rotation with identity.
 ******************************************************************************/
static void safeNormalizeBodyRotation(dBodyID body) {
    const dReal *R = dBodyGetRotation(body);
    if (isBadRotation(R)) {
        dMatrix3 identity;
        dRSetIdentity(identity);
        dBodySetRotation(body, identity);
    }
}

/******************************************************************************
 * drawText: Draw 2D text.
 ******************************************************************************/
static void drawText(const char* text, float x, float y)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, 640, 0, 480);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRasterPos2f(x, y);
    for (const char* c = text; *c != '\0'; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
    }
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

/******************************************************************************
 * drawObstacleLines: Draw green lines on the ground for obstacle avoidance.
 ******************************************************************************/
static void drawObstacleLines(void)
{
    glDisable(GL_LIGHTING);
    glLineWidth(8.0);
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    for (int i = 0; i < totalObstaclesCreated; i++) {
        const dReal* pos = dGeomGetPosition(obstacles[i]);
        double sy = obstacleSizes[i][1] / 2.0;
        double x = pos[0] + 0.5;
        double y1 = pos[1] - sy - 2.0;
        double y2 = pos[1] + sy + 2.0;
        glVertex3f(x, y1, 0.0);
        glVertex3f(x, y2, 0.0);
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

/******************************************************************************
 * createBox: Create a box body.
 ******************************************************************************/
static dBodyID createBox(dReal length, dReal width, dReal height, dReal massValue, dGeomID *geomOut)
{
    dMass m;
    dBodyID body = dBodyCreate(world);
    dMassSetBoxTotal(&m, massValue, length, width, height);
    dBodySetMass(body, &m);
    dGeomID geom = dCreateBox(space, length, width, height);
    dGeomSetBody(geom, body);
    if (geomOut)
        *geomOut = geom;
    return body;
}

/******************************************************************************
 * createCylinder: Create a cylinder body.
 ******************************************************************************/
static dBodyID createCylinder(dReal radius, dReal length, dReal massValue, bool isWheel, dGeomID *wheelTransformOut)
{
    dMass m;
    dBodyID body = dBodyCreate(world);
    dMassSetCylinderTotal(&m, massValue, 3, radius, length);
    dBodySetMass(body, &m);
    if (isWheel) {
        dGeomID cylinderGeom = dCreateCylinder(0, radius, length);
        dGeomID transGeom = dCreateGeomTransform(space);
        dGeomTransformSetGeom(transGeom, cylinderGeom);
        dGeomSetBody(transGeom, body);
        dMatrix3 R;
        dRFromAxisAndAngle(R, 1, 0, 0, M_PI/2);
        dGeomSetRotation(transGeom, R);
        if (wheelTransformOut)
            *wheelTransformOut = transGeom;
    } else {
        dGeomID geom = dCreateCylinder(space, radius, length);
        dGeomSetBody(geom, body);
    }
    return body;
}

/******************************************************************************
 * createQuadruped: Create a quadruped.
 ******************************************************************************/
static void createQuadruped(Quadruped *quad, double x, double y)
{
    double anchorX[4], anchorY[4];
    for (int i = 0; i < 4; i++) {
        double signX = (i < 2) ? -1.0 : 1.0;
        double signY = ((i % 2)==0) ? -1.0 : 1.0;
        anchorX[i] = x + signX*(BODY_LENGTH*0.5);
        anchorY[i] = y + signY*(BODY_WIDTH*0.5);
    }
    double baseHeight = 0.0;
    for (int i = 0; i < totalObstaclesCreated; i++) {
        const dReal* pos = dGeomGetPosition(obstacles[i]);
        double sx = obstacleSizes[i][0];
        double sy = obstacleSizes[i][1];
        double half_sx = sx/2.0;
        double half_sy = sy/2.0;
        if (x >= pos[0]-half_sx && x <= pos[0]+half_sx &&
            y >= pos[1]-half_sy && y <= pos[1]+half_sy) {
            double top = pos[2] + (obstacleSizes[i][2]/2.0);
            if (top > baseHeight)
                baseHeight = top;
        }
    }
    double hipZ = baseHeight + LEG_LENGTH + 0.1;
    double torsoZ = hipZ + (BODY_HEIGHT*0.5);
    
    quad->body = createBox(BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT, BODY_MASS, &quad->bodyGeom);
    dBodySetPosition(quad->body, x, y, torsoZ);
    dBodySetData(quad->body, quad);
    quad->fitness = x;
    
    quad->spawnX = x;
    quad->spawnTime = simulationTime;
    quad->prevX = x;
    quad->prevY = y;
    quad->lastMoveTime = simulationTime;
    quad->stagnancyPunished = false;
    quad->respawnCount = 0;
    quad->lastRespawnTime = simulationTime;
    quad->blinkType = BLINK_NONE;
    quad->blinkCount = 0;
    quad->blinkStartTime = 0.0;
    quad->collisionPenalty = 0.0;
    
    const dReal* tPos = dBodyGetPosition(targetBall);
    double dx = tPos[0]-x;
    double dy = tPos[1]-y;
    double dz = tPos[2]-torsoZ;
    double dist = sqrt(dx*dx+dy*dy+dz*dz);
    quad->distanceToTarget = dist;
    quad->prevTargetDistance = dist;
    quad->prevCriticValue = 0.0;
    
    for (int i = 0; i < NUM_RAYS; i++) {
        quad->raySensors[i] = dCreateRay(space, SENSOR_MAX_DIST);
        dGeomRaySet(quad->raySensors[i], x, y, torsoZ, 1, 0, 0);
        quad->sensorValues[i] = 1.0;
    }
    
    for (int i = 0; i < 4; i++){
        quad->leg[i] = createBox(LEG_WIDTH, LEG_WIDTH, LEG_LENGTH, LEG_MASS, &quad->legGeom[i]);
        dBodySetPosition(quad->leg[i], anchorX[i], anchorY[i], hipZ - LEG_LENGTH*0.5);
        dBodySetData(quad->leg[i], quad);
        
        quad->hip[i] = dJointCreateHinge(world, 0);
        dJointAttach(quad->hip[i], quad->body, quad->leg[i]);
        dJointSetHingeAnchor(quad->hip[i], anchorX[i], anchorY[i], hipZ);
        dJointSetHingeAxis(quad->hip[i], 0, 1, 0);
        dJointSetHingeParam(quad->hip[i], dParamFMax, HIP_FMAX);
        dJointSetHingeParam(quad->hip[i], dParamERP, 0.8);
        dJointSetHingeParam(quad->hip[i], dParamCFM, 1e-5);
        if (i < 2) {
            dJointSetHingeParam(quad->hip[i], dParamLoStop, -1.0);
            dJointSetHingeParam(quad->hip[i], dParamHiStop, -1.0);
        } else {
            dJointSetHingeParam(quad->hip[i], dParamLoStop, 1.0);
            dJointSetHingeParam(quad->hip[i], dParamHiStop, 1.0);
        }
        
        quad->wheel[i] = createCylinder(WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS, true, &quad->wheelTransform[i]);
        double offsetY = ((i % 2)==0) ? -(LEG_WIDTH/2.0+WHEEL_RADIUS) : (LEG_WIDTH/2.0+WHEEL_RADIUS);
        dBodySetPosition(quad->wheel[i], anchorX[i], anchorY[i]+offsetY, hipZ - LEG_LENGTH);
        dBodySetData(quad->wheel[i], quad);
        
        quad->wheelJoint[i] = dJointCreateHinge(world, 0);
        dJointAttach(quad->wheelJoint[i], quad->leg[i], quad->wheel[i]);
        dJointSetHingeAnchor(quad->wheelJoint[i], anchorX[i], anchorY[i]+offsetY, hipZ - LEG_LENGTH);
        dJointSetHingeAxis(quad->wheelJoint[i], 0, 1, 0);
        dJointSetHingeParam(quad->wheelJoint[i], dParamLoStop, -dInfinity);
        dJointSetHingeParam(quad->wheelJoint[i], dParamHiStop, dInfinity);
        dJointSetHingeParam(quad->wheelJoint[i], dParamFMax, WHEEL_FMAX);
        dJointSetHingeParam(quad->wheelJoint[i], dParamERP, 80);
        dJointSetHingeParam(quad->wheelJoint[i], dParamCFM, 1e-4);
    }
}

/******************************************************************************
 * createStaticObstacles: Create obstacles and boundaries.
 ******************************************************************************/
void createStaticObstacles(int totalObstacles, double innerMargin, double outerMargin)
{
    totalObstaclesCreated = 0;
    for (int i = 0; i < totalObstacles; i++) {
        double x = -60.0 + innerMargin + ((double)rand()/RAND_MAX) * ((60.0 - outerMargin) - (-60.0 + innerMargin));
        double y = -70.0 + innerMargin + ((double)rand()/RAND_MAX) * ((70.0 - outerMargin) - (-70.0 + innerMargin));
        double sx = 0.5 + ((double)rand()/RAND_MAX) * 1.0;
        double sy = 0.5 + ((double)rand()/RAND_MAX) * 1.0;
        double sz = 0.1 + ((double)rand()/RAND_MAX) * (2.0 - 0.1);
        double pz = sz / 2.0;
        
        dGeomID geom = dCreateBox(space, sx, sy, sz);
        dGeomSetPosition(geom, x, y, pz);
        dGeomSetData(geom, (void*)"obstacle");
        obstacles[totalObstaclesCreated] = geom;
        obstacleSizes[totalObstaclesCreated][0] = sx;
        obstacleSizes[totalObstaclesCreated][1] = sy;
        obstacleSizes[totalObstaclesCreated][2] = sz;
        totalObstaclesCreated++;
        if (totalObstaclesCreated >= MAX_OBSTACLES)
            return;
    }
}

static void drawObstacles(void)
{
    dsSetTexture(DS_SKY);
    dsSetColor(1.0, 1.0, 1.0);
    for (int i = 0; i < totalObstaclesCreated; i++) {
        const dReal* pos = dGeomGetPosition(obstacles[i]);
        dsDrawBox(pos, dGeomGetRotation(obstacles[i]), obstacleSizes[i]);
    }
}

/******************************************************************************
 * initODE: Initialize ODE world and boundaries.
 ******************************************************************************/
void initODE(void)
{
    dInitODE();
    world = dWorldCreate();
    space = dHashSpaceCreate(0);
    contactGroup = dJointGroupCreate(0);
    dWorldSetGravity(world, 0, 0, -9.81);
    dWorldSetCFM(world, 1e-5);
    dWorldSetERP(world, 0.2);
    dWorldSetQuickStepNumIterations(world, 50);
    groundPlaneGeom = dCreatePlane(space, 0, 0, 1, 0);
    
    boundaryBoxes[0] = dCreateBox(space, (60.0 - (-80.0)) + 1.0, 1.0, 5.0);
    dGeomSetPosition(boundaryBoxes[0], (60.0 + (-80.0))/2.0, 70.0 + 0.5, 5.0/2.0);
    dGeomSetData(boundaryBoxes[0], (void*)SIDE_WALL);
    
    boundaryBoxes[1] = dCreateBox(space, (60.0 - (-80.0)) + 1.0, 1.0, 5.0);
    dGeomSetPosition(boundaryBoxes[1], (60.0 + (-80.0))/2.0, -70.0 - 0.5, 5.0/2.0);
    dGeomSetData(boundaryBoxes[1], (void*)SIDE_WALL);
    
    boundaryBoxes[2] = dCreateBox(space, 1.0, (70.0 - (-70.0)) + 1.0, 5.0);
    dGeomSetPosition(boundaryBoxes[2], -80.0 - 0.5, 0.0, 5.0/2.0);
    dGeomSetData(boundaryBoxes[2], (void*)SPAWN_WALL);
    
    boundaryBoxes[3] = dCreateBox(space, 1.0, (70.0 - (-70.0)) + 1.0, 5.0);
    dGeomSetPosition(boundaryBoxes[3], 60.0 + 0.5, 0.0, 5.0/2.0);
    dGeomSetData(boundaryBoxes[3], (void*)FAR_WALL);
    
    double targetRadius = 2.0;
    targetBall = dBodyCreate(world);
    dMass m;
    dMassSetSphere(&m, 2.0, targetRadius);
    dBodySetMass(targetBall, &m);
    targetBallGeom = dCreateSphere(space, targetRadius);
    dGeomSetBody(targetBallGeom, targetBall);
    dBodySetPosition(targetBall, 55.0, 0.0, 20.0);
}

/******************************************************************************
 * cleanupODE: Clean up ODE resources.
 ******************************************************************************/
void cleanupODE(void)
{
    dJointGroupDestroy(contactGroup);
    dSpaceDestroy(space);
    dWorldDestroy(world);
    dCloseODE();
}

/******************************************************************************
 * getEnvironmentHeight: Return environment height at a point.
 ******************************************************************************/
static double getEnvironmentHeight(double x, double y)
{
    double height = 0.0;
    for (int i = 0; i < totalObstaclesCreated; i++) {
        const dReal* pos = dGeomGetPosition(obstacles[i]);
        double sx = obstacleSizes[i][0];
        double sy = obstacleSizes[i][1];
        double half_sx = sx / 2.0;
        double half_sy = sy / 2.0;
        if (x >= pos[0]-half_sx && x <= pos[0]+half_sx &&
            y >= pos[1]-half_sy && y <= pos[1]+half_sy) {
            double top = pos[2] + (obstacleSizes[i][2] / 2.0);
            if (top > height)
                height = top;
        }
    }
    return height;
}

/******************************************************************************
 * replaceQuadruped: Replace a quadruped.
 ******************************************************************************/
static void replaceQuadruped(int index)
{
    Quadruped *quad = &wobjects.quads[index];
    if (quad->body) { dBodyDestroy(quad->body); quad->body = 0; }
    if (quad->bodyGeom) { dGeomDestroy(quad->bodyGeom); quad->bodyGeom = 0; }
    for (int j = 0; j < 4; j++) {
        if (quad->leg[j]) { dBodyDestroy(quad->leg[j]); quad->leg[j] = 0; }
        if (quad->legGeom[j]) { dGeomDestroy(quad->legGeom[j]); quad->legGeom[j] = 0; }
        if (quad->hip[j]) { dJointDestroy(quad->hip[j]); quad->hip[j] = 0; }
        if (quad->wheelJoint[j]) { dJointDestroy(quad->wheelJoint[j]); quad->wheelJoint[j] = 0; }
        if (quad->wheel[j]) { dBodyDestroy(quad->wheel[j]); quad->wheel[j] = 0; }
        if (quad->wheelTransform[j]) { dGeomDestroy(quad->wheelTransform[j]); quad->wheelTransform[j] = 0; }
    }
    for (int k = 0; k < NUM_RAYS; k++) {
        if (quad->raySensors[k]) { dGeomDestroy(quad->raySensors[k]); quad->raySensors[k] = 0; }
    }
    quad->respawnCount++;
    quad->lastRespawnTime = simulationTime;
    
    // Heavy penalty for replacement.
    double dummy[ACTOR_OUTPUTS];
    runNeuralNetwork(quad, -100.0, dummy);
    quad->fitness -= 200.0;
    
    double margin = 2.0;
    double startX = -80.0 + margin;
    double startY = -70.0 + margin;
    double endY = 80.0 - margin;
    double spacing = (NUM_QUADRUPEDS > 1) ? (endY - startY) / (NUM_QUADRUPEDS - 1) : 0;
    double x = startX;
    double y = startY + index * spacing;
    createQuadruped(quad, x, y);
}

/******************************************************************************
 * nearCallback: Collision callback.
 ******************************************************************************/
static void nearCallback(void *data, dGeomID o1, dGeomID o2)
{
    // Skip collisions involving rays.
    if (dGeomGetClass(o1) == dRayClass || dGeomGetClass(o2) == dRayClass)
        return;
    
    dContact contacts[8];
    int n = dCollide(o1, o2, 8, &contacts[0].geom, sizeof(dContact));
    for (int i = 0; i < n; i++) {
        contacts[i].surface.mode = dContactBounce | dContactSoftERP | dContactSoftCFM;
        contacts[i].surface.mu = 1200.0;
        contacts[i].surface.bounce = 0.7;
        contacts[i].surface.soft_erp = 1.0;
        contacts[i].surface.soft_cfm = 1e-4;
        dJointID c = dJointCreateContact(world, contactGroup, &contacts[i]);
        dJointAttach(c, dGeomGetBody(o1), dGeomGetBody(o2));
        
        void* data1 = dGeomGetData(o1);
        void* data2 = dGeomGetData(o2);
        
        // --- Handle collisions with obstacles ---
        if (data1 == (void*)"obstacle") {
            dBodyID b = dGeomGetBody(o2);
            if (b) {
                Quadruped *quad = (Quadruped*)dBodyGetData(b);
                if (quad)
                    quad->collisionPenalty -= 5.0;
            }
        }
        else if (data2 == (void*)"obstacle") {
            dBodyID b = dGeomGetBody(o1);
            if (b) {
                Quadruped *quad = (Quadruped*)dBodyGetData(b);
                if (quad)
                    quad->collisionPenalty -= 5.0;
            }
        }
        
        // --- Handle collisions involving legs ---
        if (data1 == (void*)"leg") {
            dBodyID b = dGeomGetBody(o1);
            if (b) {
                Quadruped *quad = (Quadruped*)dBodyGetData(b);
                dBodyID other = dGeomGetBody(o2);
                if (!other || dBodyGetData(other) != quad)
                    quad->collisionPenalty -= 5.0;
            }
        }
        if (data2 == (void*)"leg") {
            dBodyID b = dGeomGetBody(o2);
            if (b) {
                Quadruped *quad = (Quadruped*)dBodyGetData(b);
                dBodyID other = dGeomGetBody(o1);
                if (!other || dBodyGetData(other) != quad)
                    quad->collisionPenalty -= 5.0;
            }
        }
        
        // --- New: Handle collisions with walls ---
        if (data1 == (void*)SIDE_WALL || data1 == (void*)SPAWN_WALL || data1 == (void*)FAR_WALL) {
            dBodyID b = dGeomGetBody(o2);
            if (b) {
                Quadruped *quad = (Quadruped*)dBodyGetData(b);
                if (quad)
                    quad->collisionPenalty -= 5.0;
            }
        }
        else if (data2 == (void*)SIDE_WALL || data2 == (void*)SPAWN_WALL || data2 == (void*)FAR_WALL) {
            dBodyID b = dGeomGetBody(o1);
            if (b) {
                Quadruped *quad = (Quadruped*)dBodyGetData(b);
                if (quad)
                    quad->collisionPenalty -= 5.0;
            }
        }
    }
}

/******************************************************************************
 * hasFallen: Check if quadruped has fallen.
 ******************************************************************************/
static int hasFallen(Quadruped *quad)
{
    const dReal *R = dBodyGetRotation(quad->body);
    double up_z = R[10];
    return (up_z < 0.6);
}

/******************************************************************************
 * updateSensorsAndControl: Update sensors, compute reward, and control.
 ******************************************************************************/
static void updateSensorsAndControl(Quadruped *quad)
{
    const dReal *bodyPos = dBodyGetPosition(quad->body);
    
    // Update sensor rays.
    for (int i = 0; i < SENSOR_RAY_GRID; i++) {
        for (int j = 0; j < SENSOR_RAY_GRID; j++) {
            int idx = i * SENSOR_RAY_GRID + j;
            double horiz = (((double)j - (SENSOR_RAY_GRID - 1) / 2.0) * (SENSOR_CONE_ANGLE / (SENSOR_RAY_GRID - 1)));
            double vert = (((double)i - (SENSOR_RAY_GRID - 1) / 2.0) * (SENSOR_CONE_ANGLE / (SENSOR_RAY_GRID - 1)));
            double cosVert = cos(vert), sinVert = sin(vert);
            double cosHoriz = cos(horiz), sinHoriz = sin(horiz);
            dVector3 rayDir;
            const dReal *bodyR = dBodyGetRotation(quad->body);
            dVector3 forward = { bodyR[0], bodyR[4], bodyR[8] };
            dVector3 right   = { bodyR[1], bodyR[5], bodyR[9] };
            dVector3 up      = { bodyR[2], bodyR[6], bodyR[10] };
            rayDir[0] = cosVert * cosHoriz * forward[0] + cosVert * sinHoriz * right[0] + sinVert * up[0];
            rayDir[1] = cosVert * cosHoriz * forward[1] + cosVert * sinHoriz * right[1] + sinVert * up[1];
            rayDir[2] = cosVert * cosHoriz * forward[2] + cosVert * sinHoriz * right[2] + sinVert * up[2];
            double sensorX = bodyPos[0] + forward[0] * (BODY_LENGTH * 0.5);
            double sensorY = bodyPos[1] + forward[1] * (BODY_LENGTH * 0.5);
            double sensorZ = bodyPos[2];
            dGeomRaySet(quad->raySensors[idx], sensorX, sensorY, sensorZ, rayDir[0], rayDir[1], rayDir[2]);
            
            double minReading = 1.0;
            dContactGeom contact;
            int n = dCollide(quad->raySensors[idx], groundPlaneGeom, 1, &contact, sizeof(dContactGeom));
            if (n > 0) {
                double reading = contact.depth / SENSOR_MAX_DIST;
                if (reading < minReading)
                    minReading = reading;
            }
            for (int obs = 0; obs < totalObstaclesCreated; obs++) {
                n = dCollide(quad->raySensors[idx], obstacles[obs], 1, &contact, sizeof(dContactGeom));
                if (n > 0) {
                    double reading = contact.depth / SENSOR_MAX_DIST;
                    if (reading < minReading)
                        minReading = reading;
                }
            }
            if (minReading > 1.0)
                minReading = 1.0;
            quad->sensorValues[idx] = minReading;
        }
    }
    
    // Compute current target distance.
    const dReal* targetPos = dBodyGetPosition(targetBall);
    double dx = targetPos[0] - bodyPos[0];
    double dy = targetPos[1] - bodyPos[1];
    double dz = targetPos[2] - bodyPos[2];
    double currentTargetDistance = sqrt(dx * dx + dy * dy + dz * dz);
    quad->distanceToTarget = currentTargetDistance;
    
    double reward = 0.0;
    
    // Reward for advancing at least 3 units closer.
    if ((quad->prevTargetDistance - currentTargetDistance) >= 3.0) {
        reward += 5000.0;
    }
    else if ((quad->prevTargetDistance - currentTargetDistance) >= 1.5) {
        reward += 2500.0;
    }
    else if ((quad->prevTargetDistance - currentTargetDistance) >= 0.5) {
        reward += 500.0;
    }
    else { 
        reward -= 500.0;
    }
    quad->prevTargetDistance = currentTargetDistance;
    
    // Reward for crossing an obstacle line.
    for (int i = 0; i < totalObstaclesCreated; i++) {
        const dReal* obsPos = dGeomGetPosition(obstacles[i]);
        double lineX = obsPos[0] + 0.5;
        if ((quad->prevX < lineX && bodyPos[0] >= lineX) ||
            (quad->prevX > lineX && bodyPos[0] <= lineX))
            reward += 500.0;
    }
    quad->prevX = bodyPos[0];
    quad->prevY = bodyPos[1];
    
    // Check collision penalty.
    if (quad->collisionPenalty < 0) {
        reward = -500.0;
        quad->collisionPenalty = 0.0;
    }
    
    // Blinking feedback.
    if (quad->blinkCount == 0) {
        if (reward >= 0) {
            quad->blinkType = BLINK_GREEN;
            quad->blinkCount = 2;
            quad->blinkStartTime = simulationTime;
        } else if (reward < 0) {
            quad->blinkType = BLINK_RED;
            quad->blinkCount = 2;
            quad->blinkStartTime = simulationTime;
        }
    }
    
    // Run neural network with the computed reward.
    double actions[ACTOR_OUTPUTS];
    runNeuralNetwork(quad, reward, actions);
    
    // Select the action with the highest output.
    int state = 0;
    double maxVal = actions[0];
    for (int i = 1; i < ACTOR_OUTPUTS; i++) {
        if (actions[i] > maxVal) { maxVal = actions[i]; state = i; }
    }
    
    // Execute the chosen action.
    switch (state) {
        case 0:
            for (int i = 0; i < 4; i++) {
                dJointSetHingeParam(quad->wheelJoint[i], dParamVel, -40);
            }
            break;
        case 1:
            dJointSetHingeParam(quad->wheelJoint[0], dParamVel, -40);
            dJointSetHingeParam(quad->wheelJoint[2], dParamVel, -40);
            dJointSetHingeParam(quad->wheelJoint[1], dParamVel, 10);
            dJointSetHingeParam(quad->wheelJoint[3], dParamVel, 10);
            break;
        case 2:
            dJointSetHingeParam(quad->wheelJoint[0], dParamVel, 40);
            dJointSetHingeParam(quad->wheelJoint[2], dParamVel, 40);
            dJointSetHingeParam(quad->wheelJoint[1], dParamVel, -10);
            dJointSetHingeParam(quad->wheelJoint[3], dParamVel, -10);
            break;
        case 3:
            for (int i = 0; i < 4; i++) {
                dJointSetHingeParam(quad->wheelJoint[i], dParamVel, 30);
            }
            break;
        default:
            break;
    }
}

/******************************************************************************
 * simLoop: Main simulation loop.
 ******************************************************************************/
static void simLoop(int pause)
{
    if (!pause) {
        simulationTime += SIMULATION_DT;
        for (int i = 0; i < NUM_QUADRUPEDS; i++) {
            Quadruped *quad = &wobjects.quads[i];
            if (!quad->body) continue;
            updateSensorsAndControl(quad);
            const dReal *pos = dBodyGetPosition(quad->body);
            quad->fitness = pos[0];
            if (hasFallen(quad)) {
                double dummy[ACTOR_OUTPUTS];
                runNeuralNetwork(quad, -100.0, dummy);
                replaceQuadruped(i);
                continue;
            }
            const dReal *bodyR = dBodyGetRotation(quad->body);
            dVector3 forward = { bodyR[0], bodyR[4], bodyR[8] };
            if (fabs(forward[0]) < SIDE_WALL_THRESHOLD) {
                double dummy[ACTOR_OUTPUTS];
                runNeuralNetwork(quad, -100.0, dummy);
                replaceQuadruped(i);
                continue;
            }
        }
        
        // Reshuffle logic.
        bool reshuffleTriggered = false;
        for (int i = 0; i < NUM_QUADRUPEDS; i++) {
            Quadruped *quad = &wobjects.quads[i];
            if (quad->respawnCount >= RESHUFFLE_RESPAWN_THRESHOLD &&
                (simulationTime - quad->lastRespawnTime) < RESHUFFLE_TIME_WINDOW) {
                reshuffleTriggered = true;
                break;
            }
        }
        if (reshuffleTriggered) {
            for (int i = 0; i < NUM_QUADRUPEDS; i++) {
                Quadruped *quad = &wobjects.quads[i];
                if (quad->body) { dBodyDestroy(quad->body); quad->body = 0; }
                if (quad->bodyGeom) { dGeomDestroy(quad->bodyGeom); quad->bodyGeom = 0; }
                for (int j = 0; j < 4; j++) {
                    if (quad->leg[j]) { dBodyDestroy(quad->leg[j]); quad->leg[j] = 0; }
                    if (quad->legGeom[j]) { dGeomDestroy(quad->legGeom[j]); quad->legGeom[j] = 0; }
                    if (quad->hip[j]) { dJointDestroy(quad->hip[j]); quad->hip[j] = 0; }
                    if (quad->wheelJoint[j]) { dJointDestroy(quad->wheelJoint[j]); quad->wheelJoint[j] = 0; }
                    if (quad->wheel[j]) { dBodyDestroy(quad->wheel[j]); quad->wheel[j] = 0; }
                    if (quad->wheelTransform[j]) { dGeomDestroy(quad->wheelTransform[j]); quad->wheelTransform[j] = 0; }
                }
                for (int k = 0; k < NUM_RAYS; k++) {
                    if (quad->raySensors[k]) { dGeomDestroy(quad->raySensors[k]); quad->raySensors[k] = 0; }
                }
                quad->respawnCount = 0;
                quad->lastRespawnTime = simulationTime;
                quad->blinkCount = 4;
                quad->blinkStartTime = simulationTime;
            }
            initNetwork();
            double margin = 2.0;
            double startX = -60.0 + margin;
            double startY = -25.0 + margin;
            double endY = 25.0 - margin;
            double spacing = (NUM_QUADRUPEDS > 1) ? (endY - startY) / (NUM_QUADRUPEDS - 1) : 0;
            for (int i = 0; i < NUM_QUADRUPEDS; i++) {
                double x = startX;
                double y = startY + i * spacing;
                createQuadruped(&wobjects.quads[i], x, y);
            }
        }
        
        // Global weight-change blink.
        if (globalWeightChangeBlink > 0) {
            for (int i = 0; i < NUM_QUADRUPEDS; i++) {
                wobjects.quads[i].blinkType = BLINK_BLUE;
                wobjects.quads[i].blinkCount = 2;
                wobjects.quads[i].blinkStartTime = simulationTime;
            }
            globalWeightChangeBlink = 0.0;
        }
        
        dSpaceCollide(space, 0, &nearCallback);
        dWorldQuickStep(world, SIMULATION_DT);
        dJointGroupEmpty(contactGroup);
        for (int i = 0; i < NUM_QUADRUPEDS; i++) {
            Quadruped *quad = &wobjects.quads[i];
            if (quad->body)
                safeNormalizeBodyRotation(quad->body);
        }
        safeNormalizeBodyRotation(targetBall);
    }
    
    #if USE_TEXTURE
    dsSetTexture(DS_CHECKERED);
    #endif
    dsSetColor(1.0, 1.0, 1.0);
    float groundSize[3] = {10.0f, 10.0f, 0.01f};
    float groundPos[3]  = {0.0f, 0.0f, 0.0f};
    float groundRot[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    dsDrawBox(groundPos, groundRot, groundSize);
    dsSetTexture(DS_GROUND);
    
    for (int i = 0; i < 4; i++) {
        const dReal* pos = dGeomGetPosition(boundaryBoxes[i]);
        const dReal* rot = dGeomGetRotation(boundaryBoxes[i]);
        dVector3 dims;
        if (i < 2) {
            dims[0] = (60.0 - (-80.0)) + 1.0;
            dims[1] = 1.0;
            dims[2] = 5.0;
        } else {
            dims[0] = 1.0;
            dims[1] = (70.0 - (-70.0)) + 1.0;
            dims[2] = 5.0;
        }
        dsDrawBox(pos, rot, dims);
    }
    
    drawObstacles();
    drawObstacleLines();
    
    dsSetTexture(DS_SKY);
    dsSetColor(0.0, 1.0, 0.0);
    const dReal* tPos = dBodyGetPosition(targetBall);
    dsDrawSphere(tPos, dBodyGetRotation(targetBall), 2.0);
    
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glLineWidth(1.0);
    glBegin(GL_LINES);
    for (int q = 0; q < NUM_QUADRUPEDS; q++) {
        Quadruped *quad = &wobjects.quads[q];
        if (!quad->body) continue;
        const dReal *bodyPos = dBodyGetPosition(quad->body);
        const dReal *bodyR = dBodyGetRotation(quad->body);
        dVector3 forward = { bodyR[0], bodyR[4], bodyR[8] };
        dVector3 right = { bodyR[1], bodyR[5], bodyR[9] };
        dVector3 up = { bodyR[2], bodyR[6], bodyR[10] };
        for (int i = 0; i < SENSOR_RAY_GRID; i++) {
            for (int j = 0; j < SENSOR_RAY_GRID; j++) {
                int idx = i * SENSOR_RAY_GRID + j;
                double horiz = (((double)j - (SENSOR_RAY_GRID - 1) / 2.0) * (SENSOR_CONE_ANGLE / (SENSOR_RAY_GRID - 1)));
                double vert = (((double)i - (SENSOR_RAY_GRID - 1) / 2.0) * (SENSOR_CONE_ANGLE / (SENSOR_RAY_GRID - 1)));
                double cosVert = cos(vert), sinVert = sin(vert);
                double cosHoriz = cos(horiz), sinHoriz = sin(horiz);
                dVector3 rayDir;
                rayDir[0] = cosVert * cosHoriz * forward[0] + cosVert * sinHoriz * right[0] + sinVert * up[0];
                rayDir[1] = cosVert * cosHoriz * forward[1] + cosVert * sinHoriz * right[1] + sinVert * up[1];
                rayDir[2] = cosVert * cosHoriz * forward[2] + cosVert * sinHoriz * right[2] + sinVert * up[2];
                double sensorX = bodyPos[0] + forward[0] * (BODY_LENGTH * 0.5);
                double sensorY = bodyPos[1] + forward[1] * (BODY_LENGTH * 0.5);
                double sensorZ = bodyPos[2];
                double reading = quad->sensorValues[idx];
                double drawLength = DRAWN_MAX_LENGTH * reading;
                float r = (float)(1.0 - reading);
                float b = (float)(reading);
                float g = 0.2f;  
                glColor3f(r, g, b);
                glVertex3d(sensorX, sensorY, sensorZ);
                glVertex3d(sensorX + rayDir[0] * drawLength,
                           sensorY + rayDir[1] * drawLength,
                           sensorZ + rayDir[2] * drawLength);
            }
        }
    }
    glEnd();
    glPopAttrib();
    
    for (int i = 0; i < NUM_QUADRUPEDS; i++) {
        Quadruped *quad = &wobjects.quads[i];
        if (quad->body) {
            const dReal *bodyPos = dBodyGetPosition(quad->body);
            float color[3] = {1.0f, 1.0f, 1.0f};
            double elapsed = simulationTime - quad->blinkStartTime;
            if (quad->blinkCount > 0 && elapsed < quad->blinkCount * BLINK_PERIOD) {
                if (fmod(elapsed, BLINK_PERIOD) < (BLINK_PERIOD / 2)) {
                    switch(quad->blinkType) {
                        case BLINK_GREEN:
                            color[0] = 0.0f; color[1] = 1.0f; color[2] = 0.0f;
                            break;
                        case BLINK_RED:
                            color[0] = 1.0f; color[1] = 0.0f; color[2] = 0.0f;
                            break;
                        case BLINK_BLUE:
                            color[0] = 0.0f; color[1] = 0.0f; color[2] = 1.0f;
                            break;
                        default:
                            break;
                    }
                }
            } else if (quad->blinkCount > 0) {
                quad->blinkCount = 0;
                quad->blinkType = BLINK_NONE;
            }
            dsSetColor(color[0], color[1], color[2]);
            dsSetTexture(DS_WOOD);
            dVector3 sides = {BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT};
            dsDrawBox(bodyPos, dBodyGetRotation(quad->body), sides);
            for (int j = 0; j < 4; j++) {
                if (quad->leg[j]) {
                    dVector3 legSides = {LEG_WIDTH, LEG_WIDTH, LEG_LENGTH};
                    dsDrawBox(dBodyGetPosition(quad->leg[j]), dBodyGetRotation(quad->leg[j]), legSides);
                }
                if (quad->wheel[j]) {
                    dsDrawCylinder(dBodyGetPosition(quad->wheel[j]), dBodyGetRotation(quad->wheel[j]), WHEEL_RADIUS, WHEEL_WIDTH);
                }
            }
        }
    }
    
    {
        char textBuffer[64];
        int countInside = 0;
        for (int i = 0; i < NUM_QUADRUPEDS; i++) {
            Quadruped *quad = &wobjects.quads[i];
            if (quad->body) {
                const dReal* pos = dBodyGetPosition(quad->body);
                if (fabs(pos[0]) <= 60.0 && fabs(pos[1]) <= 35.0)
                    countInside++;
            }
        }
        sprintf(textBuffer, "Quadrupeds in Course: %d", countInside);
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_LIGHTING);
        drawText(textBuffer, 10.0f, 10.0f);
        glPopAttrib();
    }
}

/******************************************************************************
 * start: Setup initial view.
 ******************************************************************************/
static void start(void)
{
    const dReal *quadPos = dBodyGetPosition(wobjects.quads[0].body);
    float xyz[3] = { quadPos[0] + 3.0f, quadPos[1] - 2.0f, quadPos[2] + 2.5f };
    float hpr[3] = {90.0f, -15.0f, 0.0f};
    dsSetViewpoint(xyz, hpr);
    printf("Press 'p' to pause, 'q' to quit.\n");
}

/******************************************************************************
 * command: Keyboard callback.
 ******************************************************************************/
static void command(int cmd)
{
    if (cmd == 'q')
        dsStop();
}

/******************************************************************************
 * stop: Stop callback.
 ******************************************************************************/
static void stop(void)
{
}

/******************************************************************************
 * randomUniform: Return a random double between min and max.
 ******************************************************************************/
static double randomUniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

/******************************************************************************
 * main: Main function.
 ******************************************************************************/
int main(int argc, char *argv[])
{
    srand((unsigned)time(NULL));
#ifdef DS_EXPORT
    dsFunctions dsFunc;
    dsFunc.version = DS_VERSION;
    dsFunc.start = &start;
    dsFunc.step = &simLoop;
    dsFunc.command = &command;
    dsFunc.stop = &stop;
    dsFunc.path_to_textures = "/Users/willnorden/Downloads/ode-0.16.6/drawstuff/textures";
#else
    dsFunctions dsFunc = { DS_VERSION, &start, &simLoop, &command, &stop,
        "/Users/willnorden/Downloads/ode-0.16.6/drawstuff/textures" };
#endif

    initODE();
    createStaticObstacles(2000, 1.0, 1.0);
    initNetwork();
    
    double margin = 2.0;
    double startX = -80.0 + margin;
    double startY = -70.0 + margin;
    double endY = 80.0 - margin;
    double spacing = (NUM_QUADRUPEDS > 1) ? (endY - startY) / (NUM_QUADRUPEDS - 1) : 0;
    for (int i = 0; i < NUM_QUADRUPEDS; i++) {
        double x = startX;
        double y = startY + i * spacing;
        createQuadruped(&wobjects.quads[i], x, y);
    }
    
    dsSimulationLoop(argc, argv, 640, 480, &dsFunc);
    cleanupODE();
    return 0;
}
