#define GL_SILENCE_DEPRECATION // silencing warning messages

#include <ode/ode.h>
#include <drawstuff/drawstuff.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <GLUT/glut.h>

// Provide a dummy for dGeomTransformUpdate if not declared.
#ifndef dGeomTransformUpdate
static inline void dGeomTransformUpdate(dGeomID g) { /* nothing */ }
#endif

#ifdef dDOUBLE
#  define dsDrawBox      dsDrawBoxD
#  define dsDrawCylinder dsDrawCylinderD
#  define dsDrawSphere   dsDrawSphereD
#endif

#ifndef DS_VERSION
#define DS_VERSION 0
#endif

#if DS_VERSION >= 2
  #define USE_SHADOWS 1
  #define USE_TEXTURE 1
#else
  #define USE_SHADOWS 0
  #define USE_TEXTURE 1
#endif

/******************************************************************************
 * Simulation parameters
 ******************************************************************************/
#define NUM_QUADRUPEDS    40
#define SIMULATION_DT     0.02

// Sensor parameters: sensor's cone-ray grid
#define SENSOR_RAY_GRID   5                      // 5 rays per row and column
#define NUM_RAYS          (SENSOR_RAY_GRID * SENSOR_RAY_GRID)
#define SENSOR_MAX_DIST   20.0                   // Maximum ray length (for sensing)
#define SENSOR_CONE_ANGLE (M_PI/4.0)             // Maximum angle deviation (radians)
#define DRAWN_MAX_LENGTH  2.0                    // Maximum length for drawing the ray

/******************************************************************************
 * Quadruped geometry and mass parameters
 ******************************************************************************/
#define BODY_LENGTH       0.7
#define BODY_WIDTH        0.3
#define BODY_HEIGHT       0.1
#define BODY_MASS         50.0

#define LEG_WIDTH         0.04
#define LEG_LENGTH        0.5
#define LEG_MASS          30.0

#define WHEEL_RADIUS      0.1                    
#define WHEEL_WIDTH       0.1                    
#define WHEEL_MASS        20.0

#define HIP_FMAX          5000.0                 
#define WHEEL_FMAX        5000.0
#define MAX_OBSTACLES     10000

/******************************************************************************
 * Neural Network & RL parameters (shared by all quadrupeds)
 ******************************************************************************/
#define NN_HIDDEN         20                     // Number of neurons in each hidden layer.
#define NN_OUTPUTS        4                      // One output per hip.
const double POLICY_SIGMA = 0.1;                 // Standard deviation for exploration noise.
const double LEARNING_RATE = 0.001;              // Learning rate for policy gradient update.

/******************************************************************************
 * Helper: Gaussian noise generator using Box–Muller.
 ******************************************************************************/
double gaussianNoise(double mean, double stddev) {
    static int haveSpare = 0;
    static double spare;
    if(haveSpare) {
        haveSpare = 0;
        return mean + stddev * spare;
    }
    haveSpare = 1;
    double u, v, s;
    do {
        u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while(s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + stddev * (u * s);
}

/******************************************************************************
 * Global shared network parameters for a 3-hidden layer network.
 *
 * Layer sizes:
 *   Input: NUM_RAYS
 *   Hidden1: NN_HIDDEN
 *   Hidden2: NN_HIDDEN
 *   Hidden3: NN_HIDDEN
 *   Output: NN_OUTPUTS (one per hip)
 ******************************************************************************/
static double global_nn_W1[NN_HIDDEN][NUM_RAYS];
static double global_nn_b1[NN_HIDDEN];

static double global_nn_W2[NN_HIDDEN][NN_HIDDEN];
static double global_nn_b2[NN_HIDDEN];

static double global_nn_W3[NN_HIDDEN][NN_HIDDEN];
static double global_nn_b3[NN_HIDDEN];

// Modified output layer: four outputs (one per hip)
static double global_nn_W4[NN_OUTPUTS][NN_HIDDEN]; 
static double global_nn_b4[NN_OUTPUTS];

/******************************************************************************
 * Data Structures
 ******************************************************************************/
typedef struct {
    dBodyID body;              // Torso
    dBodyID leg[4];            // Leg segments
    dBodyID wheel[4];          // Wheels
    dGeomID wheelTransform[4]; // Transform geoms for wheels
    dJointID hip[4];           // Hip joints (body-to-leg)
    dJointID wheelJoint[4];    // Wheel joints (leg-to-wheel)
    int hit;                   // Flag: has the quadruped been hit/disqualified?
    double fitness;            // Fitness (e.g. x position)
    
    // For RL reward calculation.
    double spawnX;
    double spawnTime;
    double prevX;
    
    // Sensor ray geoms – used solely for sensing.
    dGeomID raySensors[NUM_RAYS];
    // Stored sensor readings (normalized: 0 means obstacle very near; 1 means full length)
    double sensorValues[NUM_RAYS];
} Quadruped;

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

/******************************************************************************
 * Forward declarations
 ******************************************************************************/
static void simLoop(int pause);
static void start();
static void command(int cmd);
static void stop();
static double getEnvironmentHeight(double x, double y);
static void replaceQuadruped(int i);
static double randomUniform(double min, double max);
static void updateSensorsAndControl(Quadruped *quad);

/******************************************************************************
 * Helper: draw 2D text using GLUT bitmap fonts.
 ******************************************************************************/
void drawText(const char* text, float x, float y)
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
 * ODE collision callback.
 ******************************************************************************/
static void nearCallback(void *data, dGeomID o1, dGeomID o2)
{
    (void)data;
    if (dGeomGetClass(o1) == dRayClass || dGeomGetClass(o2) == dRayClass)
        return;

    dBodyID b1 = dGeomGetBody(o1);
    dBodyID b2 = dGeomGetBody(o2);
    if (b1 && b2 && dAreConnectedExcluding(b1, b2, dJointTypeContact))
        return;

    if ((dGeomGetData(o1) == (void*)"obstacle" && b2 && dBodyGetData(b2)) ||
        (dGeomGetData(o2) == (void*)"obstacle" && b1 && dBodyGetData(b1)) )
    {
        Quadruped *quad = (Quadruped*)(b1 ? dBodyGetData(b1) : dBodyGetData(b2));
        quad->hit = 1;
    }
    
    void *tag1 = dGeomGetData(o1);
    void *tag2 = dGeomGetData(o2);
    if ((tag1 == (void*)SPAWN_WALL || tag1 == (void*)SIDE_WALL || tag1 == (void*)FAR_WALL) &&
         b2 && dBodyGetData(b2))
    {
        Quadruped *quad = (Quadruped*)dBodyGetData(b2);
        quad->hit = 1;
    }
    if ((tag2 == (void*)SPAWN_WALL || tag2 == (void*)SIDE_WALL || tag2 == (void*)FAR_WALL) &&
         b1 && dBodyGetData(b1))
    {
        Quadruped *quad = (Quadruped*)dBodyGetData(b1);
        quad->hit = 1;
    }
    
    dContact contacts[8];
    int n = dCollide(o1, o2, 8, &contacts[0].geom, sizeof(dContact));
    for (int i = 0; i < n; i++) {
        contacts[i].surface.mode = dContactBounce | dContactSoftERP | dContactSoftCFM;
        contacts[i].surface.mu = 2000.0;
        contacts[i].surface.bounce = 0.5;
        contacts[i].surface.soft_erp = 1.0;
        contacts[i].surface.soft_cfm = 1e-4;
        dJointID c = dJointCreateContact(world, contactGroup, &contacts[i]);
        dJointAttach(c, b1, b2);
    }
}

/******************************************************************************
 * Create a box body and geom.
 ******************************************************************************/
static dBodyID createBox(dReal length, dReal width, dReal height, dReal massValue)
{
    dMass m;
    dBodyID body = dBodyCreate(world);
    dMassSetBoxTotal(&m, massValue, length, width, height);
    dBodySetMass(body, &m);
    dGeomID geom = dCreateBox(space, length, width, height);
    dGeomSetBody(geom, body);
    return body;
}

/******************************************************************************
 * Create a cylinder body and geom.
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
 * Create a single quadruped.
 ******************************************************************************/
static void createQuadruped(Quadruped *quad, double x, double y)
{
    double anchorX[4], anchorY[4];
    for (int i = 0; i < 4; i++) {
        double signX = (i < 2) ? -1.0 : 1.0;
        double signY = ((i % 2) == 0) ? -1.0 : 1.0;
        anchorX[i] = x + signX * (BODY_LENGTH * 0.5);
        anchorY[i] = y + signY * (BODY_WIDTH * 0.5);
    }
    double baseHeight = 0.0;
    for (int i = 0; i < totalObstaclesCreated; i++) {
        const dReal* pos = dGeomGetPosition(obstacles[i]);
        double sx = obstacleSizes[i][0];
        double sy = obstacleSizes[i][1];
        double half_sx = sx / 2.0;
        double half_sy = sy / 2.0;
        if (x >= pos[0] - half_sx && x <= pos[0] + half_sx &&
            y >= pos[1] - half_sy && y <= pos[1] + half_sy)
        {
            double top = pos[2] + (obstacleSizes[i][2] / 2.0);
            if (top > baseHeight)
                baseHeight = top;
        }
    }
    double hipZ = baseHeight + LEG_LENGTH + 0.1;
    double legBottomZ = hipZ - LEG_LENGTH;
    double torsoZ = hipZ + (BODY_HEIGHT * 0.5);

    quad->body = createBox(BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT, BODY_MASS);
    dBodySetPosition(quad->body, x, y, torsoZ);
    dBodySetData(quad->body, quad);
    quad->hit = 0;
    quad->fitness = x;
    
    quad->spawnX = x;
    quad->spawnTime = simulationTime;
    quad->prevX = x;
    
    for (int i = 0; i < NUM_RAYS; i++) {
        quad->raySensors[i] = dCreateRay(space, SENSOR_MAX_DIST);
        dGeomRaySet(quad->raySensors[i], x, y, torsoZ, 1, 0, 0);
        quad->sensorValues[i] = 1.0;
    }
    
    for (int i = 0; i < 4; i++) {
        quad->leg[i] = createBox(LEG_WIDTH, LEG_WIDTH, LEG_LENGTH, LEG_MASS);
        dBodySetPosition(quad->leg[i], anchorX[i], anchorY[i], hipZ - LEG_LENGTH * 0.5);
        dBodySetData(quad->leg[i], quad);
        
        quad->hip[i] = dJointCreateHinge(world, 0);
        dJointAttach(quad->hip[i], quad->body, quad->leg[i]);
        dJointSetHingeAnchor(quad->hip[i], anchorX[i], anchorY[i], hipZ);
        dJointSetHingeAxis(quad->hip[i], 0, 1, 0);
        dJointSetHingeParam(quad->hip[i], dParamFMax, HIP_FMAX);
        dJointSetHingeParam(quad->hip[i], dParamERP, 0.8);
        dJointSetHingeParam(quad->hip[i], dParamCFM, 1e-5);
        // Baseline: left legs (indices 0,1) fixed at -1.0, right legs (2,3) fixed at 1.0.
        if (i < 2) {
            dJointSetHingeParam(quad->hip[i], dParamLoStop, -1.0);
            dJointSetHingeParam(quad->hip[i], dParamHiStop, -1.0);
        } else {
            dJointSetHingeParam(quad->hip[i], dParamLoStop, 1.0);
            dJointSetHingeParam(quad->hip[i], dParamHiStop, 1.0);
        }

        quad->wheel[i] = createCylinder(WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS, true, &quad->wheelTransform[i]);
        double offsetY = ((i % 2) == 0) ? -(LEG_WIDTH / 2.0 + WHEEL_RADIUS)
                                        :  (LEG_WIDTH / 2.0 + WHEEL_RADIUS);
        dBodySetPosition(quad->wheel[i],
                         anchorX[i],
                         anchorY[i] + offsetY,
                         legBottomZ);
        dBodySetData(quad->wheel[i], quad);

        quad->wheelJoint[i] = dJointCreateHinge(world, 0);
        dJointAttach(quad->wheelJoint[i], quad->leg[i], quad->wheel[i]);
        dJointSetHingeAnchor(quad->wheelJoint[i],
                             anchorX[i],
                             anchorY[i] + offsetY,
                             legBottomZ);
        dJointSetHingeAxis(quad->wheelJoint[i], 0, 1, 0);
        dJointSetHingeParam(quad->wheelJoint[i], dParamLoStop, -dInfinity);
        dJointSetHingeParam(quad->wheelJoint[i], dParamHiStop,  dInfinity);
        dJointSetHingeParam(quad->wheelJoint[i], dParamFMax, WHEEL_FMAX);
        dJointSetHingeParam(quad->wheelJoint[i], dParamERP, 80);
        dJointSetHingeParam(quad->wheelJoint[i], dParamCFM, 1e-4);
        dJointSetHingeParam(quad->wheelJoint[i], dParamVel, -4.0); //wheel speed
    }
}

/******************************************************************************
 * Environment: obstacles and boundaries.
 ******************************************************************************/
void createStaticObstacles(int totalObstacles, double innerRadius, double outerRadius)
{
    totalObstaclesCreated = 0;
    int rings = 30;
    double radiusStep = (outerRadius - innerRadius) / rings;
    int obstaclesCreated = 0;

    for (int r = 0; r < rings && obstaclesCreated < totalObstacles; r++) {
        double radius = innerRadius + r * radiusStep;
        int obstaclesPerRing = (int)(2 * M_PI * radius / 1.0);
        if (obstaclesPerRing < 1) obstaclesPerRing = 1;
        for (int c = 0; c < obstaclesPerRing && obstaclesCreated < totalObstacles; c++) {
            double angle = (2 * M_PI / obstaclesPerRing) * c +
                           (((double)rand() / RAND_MAX) * 0.1 - 0.05);
            double randomRadOffset = (((double)rand() / RAND_MAX) * (radiusStep * 0.5)) -
                                     (radiusStep * 0.25);
            double finalRadius = radius + randomRadOffset;
            double px = finalRadius * cos(angle);
            double py = finalRadius * sin(angle);
            double sx = 0.5 + ((double)rand() / RAND_MAX) * 1.0;
            double sy = 0.5 + ((double)rand() / RAND_MAX) * 1.0;
            double sz = 0.01 + ((double)rand() / RAND_MAX) * 0.15 + ((double)rand() / RAND_MAX) * 0.35 * ((double)rand() / RAND_MAX);
            double pz = sz / 2.0;
            dGeomID geom = dCreateBox(space, sx, sy, sz);
            dGeomSetPosition(geom, px, py, pz);
            dGeomSetData(geom, (void*)"obstacle");
            obstacles[totalObstaclesCreated] = geom;
            obstacleSizes[totalObstaclesCreated][0] = sx;
            obstacleSizes[totalObstaclesCreated][1] = sy;
            obstacleSizes[totalObstaclesCreated][2] = sz;
            totalObstaclesCreated++;
            obstaclesCreated++;
            if (totalObstaclesCreated >= MAX_OBSTACLES) return;
        }
    }
}

static void drawObstacles()
{
    dsSetTexture(DS_SKY);
    dsSetColor(1.0, 1.0, 1.0);
    for (int i = 0; i < totalObstaclesCreated; i++) {
        const dReal* pos = dGeomGetPosition(obstacles[i]);
        dsDrawBox(pos, dGeomGetRotation(obstacles[i]), obstacleSizes[i]);
    }
}

/******************************************************************************
 * ODE initialization.
 ******************************************************************************/
static void initODE()
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

    boundaryBoxes[0] = dCreateBox(space, (40.0*2 + 1.0), 1.0, 5.0);
    dGeomSetPosition(boundaryBoxes[0], 0.0, 35.0 + 0.5, 5.0/2.0);
    dGeomSetData(boundaryBoxes[0], (void*)SIDE_WALL);

    boundaryBoxes[1] = dCreateBox(space, (40.0*2 + 1.0), 1.0, 5.0);
    dGeomSetPosition(boundaryBoxes[1], 0.0, -35.0 - 0.5, 5.0/2.0);
    dGeomSetData(boundaryBoxes[1], (void*)SIDE_WALL);

    boundaryBoxes[2] = dCreateBox(space, 1.0, (40.0*2), 5.0);
    dGeomSetPosition(boundaryBoxes[2], -45.0 - 0.5, 0.0, 5.0/2.0);
    dGeomSetData(boundaryBoxes[2], (void*)SPAWN_WALL);

    boundaryBoxes[3] = dCreateBox(space, 1.0, (40.0*2), 5.0);
    dGeomSetPosition(boundaryBoxes[3], 40.0 + 0.5, 0.0, 5.0/2.0);
    dGeomSetData(boundaryBoxes[3], (void*)FAR_WALL);
}

/******************************************************************************
 * Clean up ODE.
 ******************************************************************************/
static void cleanupODE()
{
    dJointGroupDestroy(contactGroup);
    dSpaceDestroy(space);
    dWorldDestroy(world);
    dCloseODE();
}

/******************************************************************************
 * Helper: Return environment height at (x,y).
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
        if (x >= pos[0] - half_sx && x <= pos[0] + half_sx &&
            y >= pos[1] - half_sy && y <= pos[1] + half_sy)
        {
            double top = pos[2] + (obstacleSizes[i][2] / 2.0);
            if (top > height)
                height = top;
        }
    }
    return height;
}

/******************************************************************************
 * Replace quadruped: destroy current bodies/joints, sensor rays, and create new ones.
 * This function reinitializes all quadrupeds.
 ******************************************************************************/
static void replaceQuadruped(int i)
{
    Quadruped *quad = &wobjects.quads[i];

    if (quad->body) { dBodyDestroy(quad->body); quad->body = 0; }
    for (int j = 0; j < 4; j++) {
        if (quad->leg[j]) { dBodyDestroy(quad->leg[j]); quad->leg[j] = 0; }
        if (quad->wheel[j]) { dBodyDestroy(quad->wheel[j]); quad->wheel[j] = 0; }
        if (quad->hip[j]) { dJointDestroy(quad->hip[j]); quad->hip[j] = 0; }
        if (quad->wheelJoint[j]) { dJointDestroy(quad->wheelJoint[j]); quad->wheelJoint[j] = 0; }
        if (quad->wheelTransform[j]) { dGeomDestroy(quad->wheelTransform[j]); quad->wheelTransform[j] = 0; }
    }
    for (int k = 0; k < NUM_RAYS; k++) {
        if (quad->raySensors[k]) { dGeomDestroy(quad->raySensors[k]); quad->raySensors[k] = 0; }
    }
    // Re-create all quadrupeds.
    double margin = 2.0;
    double startX = -45.0 + margin;
    double startY = -20.0 + margin;
    double endY   = 20.0 - margin;
    double spacing = (NUM_QUADRUPEDS > 1) ? (endY - startY) / (NUM_QUADRUPEDS - 1) : 0;
    for (int i = 0; i < NUM_QUADRUPEDS; i++) {
        double x = startX;
        double y = startY + i * spacing;
        createQuadruped(&wobjects.quads[i], x, y);
    }
}

/******************************************************************************
 * Update sensor rays and control the hip joints using the shared NN policy.
 * Hip adjustments are limited to a small delta (±0.2 radians) added to fixed
 * baselines: -1.0 for left hips (indices 0,1) and +1.0 for right hips (indices 2,3).
 ******************************************************************************/
static void updateSensorsAndControl(Quadruped *quad)
{
    const dReal *bodyPos = dBodyGetPosition(quad->body);
    const dReal *bodyR = dBodyGetRotation(quad->body);
    dVector3 forward = { bodyR[0], bodyR[4], bodyR[8] };
    dVector3 right   = { bodyR[1], bodyR[5], bodyR[9] };
    dVector3 up      = { bodyR[2], bodyR[6], bodyR[10] };

    // Adjust this value to control the downward tilt of the ray cone (in radians).
    const double pitchOffset = -0.3;  // Subtract this from vertical angle

    // Update sensor rays.
    for (int i = 0; i < SENSOR_RAY_GRID; i++) {
        for (int j = 0; j < SENSOR_RAY_GRID; j++) {
            int idx = i * SENSOR_RAY_GRID + j;
            double horiz = ((double)j - (SENSOR_RAY_GRID - 1) / 2.0) *
                           (SENSOR_CONE_ANGLE / (SENSOR_RAY_GRID - 1));
            // Apply a downward pitch offset by subtracting from the computed vertical angle.
            double vert  = (((double)i - (SENSOR_RAY_GRID - 1) / 2.0) *
                           (SENSOR_CONE_ANGLE / (SENSOR_RAY_GRID - 1))) - pitchOffset;
            double cosVert = cos(vert);
            double sinVert = sin(vert);
            double cosHoriz = cos(horiz);
            double sinHoriz = sin(horiz);
            dVector3 rayDir;
            rayDir[0] = cosVert * cosHoriz * forward[0] + cosVert * sinHoriz * right[0] + sinVert * up[0];
            rayDir[1] = cosVert * cosHoriz * forward[1] + cosVert * sinHoriz * right[1] + sinVert * up[1];
            rayDir[2] = cosVert * cosHoriz * forward[2] + cosVert * sinHoriz * right[2] + sinVert * up[2];
            double sensorX = bodyPos[0] + forward[0] * (BODY_LENGTH * 0.5);
            double sensorY = bodyPos[1] + forward[1] * (BODY_LENGTH * 0.5);
            double sensorZ = bodyPos[2];
            dGeomRaySet(quad->raySensors[idx],
                        sensorX, sensorY, sensorZ,
                        rayDir[0], rayDir[1], rayDir[2]);
                    
            dContactGeom contact;
            int n = dCollide(quad->raySensors[idx], groundPlaneGeom, 1, &contact, sizeof(dContactGeom));
            double reading = 1.0;
            if (n > 0) {
                reading = contact.depth / SENSOR_MAX_DIST;
                if (reading > 1.0) reading = 1.0;
            }
            quad->sensorValues[idx] = reading;
        }
    }
    
    // --- Shared Neural Network Forward Pass (3 hidden layers) ---
    double input[NUM_RAYS];
    for (int i = 0; i < NUM_RAYS; i++) {
        input[i] = quad->sensorValues[i];
    }
    
    double z1[NN_HIDDEN], h1[NN_HIDDEN];
    for (int i = 0; i < NN_HIDDEN; i++) {
        z1[i] = global_nn_b1[i];
        for (int j = 0; j < NUM_RAYS; j++) {
            z1[i] += global_nn_W1[i][j] * input[j];
        }
        h1[i] = (z1[i] > 0) ? z1[i] : 0.0;
    }
    
    double z2[NN_HIDDEN], h2[NN_HIDDEN];
    for (int i = 0; i < NN_HIDDEN; i++) {
        z2[i] = global_nn_b2[i];
        for (int j = 0; j < NN_HIDDEN; j++) {
            z2[i] += global_nn_W2[i][j] * h1[j];
        }
        h2[i] = (z2[i] > 0) ? z2[i] : 0.0;
    }
    
    double z3[NN_HIDDEN], h3[NN_HIDDEN];
    for (int i = 0; i < NN_HIDDEN; i++) {
        z3[i] = global_nn_b3[i];
        for (int j = 0; j < NN_HIDDEN; j++) {
            z3[i] += global_nn_W3[i][j] * h2[j];
        }
        h3[i] = (z3[i] > 0) ? z3[i] : 0.0;
    }
    
    double z4[NN_OUTPUTS], mu[NN_OUTPUTS], action[NN_OUTPUTS];
    for (int k = 0; k < NN_OUTPUTS; k++) {
        z4[k] = global_nn_b4[k];
        for (int i = 0; i < NN_HIDDEN; i++) {
            z4[k] += global_nn_W4[k][i] * h3[i];
        }
        mu[k] = tanh(z4[k]);
        double noise = gaussianNoise(0.0, POLICY_SIGMA);
        action[k] = mu[k] + noise;
    }
    
    // --- Policy Gradient Update ---
    double currentX = bodyPos[0];
    double reward = currentX - quad->prevX;
    quad->prevX = currentX;
    
    double delta4[NN_OUTPUTS];
    for (int k = 0; k < NN_OUTPUTS; k++) {
        delta4[k] = ((action[k] - mu[k]) / (POLICY_SIGMA * POLICY_SIGMA)) * (1.0 - mu[k] * mu[k]);
    }
    
    double dW4[NN_OUTPUTS][NN_HIDDEN];
    double db4[NN_OUTPUTS];
    for (int k = 0; k < NN_OUTPUTS; k++) {
        db4[k] = delta4[k];
        for (int i = 0; i < NN_HIDDEN; i++) {
            dW4[k][i] = delta4[k] * h3[i];
        }
    }
    
    double delta3[NN_HIDDEN] = {0.0};
    for (int i = 0; i < NN_HIDDEN; i++) {
        double sum = 0.0;
        for (int k = 0; k < NN_OUTPUTS; k++) {
            sum += global_nn_W4[k][i] * delta4[k];
        }
        double dReLU = (z3[i] > 0) ? 1.0 : 0.0;
        delta3[i] = sum * dReLU;
    }
    
    double dW3[NN_HIDDEN][NN_HIDDEN];
    double db3[NN_HIDDEN];
    for (int i = 0; i < NN_HIDDEN; i++) {
        db3[i] = delta3[i];
        for (int j = 0; j < NN_HIDDEN; j++) {
            dW3[i][j] = delta3[i] * h2[j];
        }
    }
    
    double delta2[NN_HIDDEN];
    for (int i = 0; i < NN_HIDDEN; i++) {
        double sum = 0.0;
        for (int k = 0; k < NN_HIDDEN; k++) {
            sum += global_nn_W3[k][i] * delta3[k];
        }
        double dReLU = (z2[i] > 0) ? 1.0 : 0.0;
        delta2[i] = sum * dReLU;
    }
    
    double dW2[NN_HIDDEN][NN_HIDDEN];
    double db2[NN_HIDDEN];
    for (int i = 0; i < NN_HIDDEN; i++) {
        db2[i] = delta2[i];
        for (int j = 0; j < NN_HIDDEN; j++) {
            dW2[i][j] = delta2[i] * h1[j];
        }
    }
    
    double delta1[NN_HIDDEN];
    for (int i = 0; i < NN_HIDDEN; i++) {
        double sum = 0.0;
        for (int k = 0; k < NN_HIDDEN; k++) {
            sum += global_nn_W2[k][i] * delta2[k];
        }
        double dReLU = (z1[i] > 0) ? 1.0 : 0.0;
        delta1[i] = sum * dReLU;
    }
    
    double dW1[NN_HIDDEN][NUM_RAYS];
    double db1[NN_HIDDEN];
    for (int i = 0; i < NN_HIDDEN; i++) {
        db1[i] = delta1[i];
        for (int j = 0; j < NUM_RAYS; j++) {
            dW1[i][j] = delta1[i] * input[j];
        }
    }
    
    for (int i = 0; i < NN_HIDDEN; i++) {
        for (int j = 0; j < NUM_RAYS; j++) {
            global_nn_W1[i][j] += LEARNING_RATE * reward * dW1[i][j];
        }
        global_nn_b1[i] += LEARNING_RATE * reward * db1[i];
    }
    for (int i = 0; i < NN_HIDDEN; i++) {
        for (int j = 0; j < NN_HIDDEN; j++) {
            global_nn_W2[i][j] += LEARNING_RATE * reward * dW2[i][j];
        }
        global_nn_b2[i] += LEARNING_RATE * reward * db2[i];
    }
    for (int i = 0; i < NN_HIDDEN; i++) {
        for (int j = 0; j < NN_HIDDEN; j++) {
            global_nn_W3[i][j] += LEARNING_RATE * reward * dW3[i][j];
        }
        global_nn_b3[i] += LEARNING_RATE * reward * db3[i];
    }
    for (int k = 0; k < NN_OUTPUTS; k++) {
        for (int i = 0; i < NN_HIDDEN; i++) {
            global_nn_W4[k][i] += LEARNING_RATE * reward * dW4[k][i];
        }
        global_nn_b4[k] += LEARNING_RATE * reward * db4[k];
    }
    
    // --- Control: Constrain hip adjustments.
    // Left hips (indices 0,1) start at -1.0, right hips (indices 2,3) at 1.0.
    // Allow only a small delta (clamped to ±0.2 radians) added to these baselines.
    for (int k = 0; k < NN_OUTPUTS; k++) {
        double delta = action[k];
        if (delta > 0.01) delta = 0.01;
        if (delta < -0.01) delta = -0.01;
        if (k < 2) {
            double desiredAngle = -1.0 + delta;
            dJointSetHingeParam(quad->hip[k], dParamLoStop, desiredAngle);
            dJointSetHingeParam(quad->hip[k], dParamHiStop, desiredAngle);
        } else {
            double desiredAngle = 1.0 + delta;
            dJointSetHingeParam(quad->hip[k], dParamLoStop, desiredAngle);
            dJointSetHingeParam(quad->hip[k], dParamHiStop, desiredAngle);
        }
    }
}

/******************************************************************************
 * Simulation loop.
 * 
 * This loop now checks the reset condition only once per second.
 ******************************************************************************/
static void simLoop(int pause)
{
    // Static variable to track the last time (in seconds) we checked the reset condition.
    static double lastResetCheck = 0.0;
    
    if (!pause) {
        simulationTime += SIMULATION_DT;
        
        // Reset hit flags for all quadrupeds.
        for (int i = 0; i < NUM_QUADRUPEDS; i++) {
            wobjects.quads[i].hit = 0;
        }
        
        int failedCount = 0;
        for (int i = 0; i < NUM_QUADRUPEDS; i++) {
            Quadruped *quad = &wobjects.quads[i];
            if (!quad->body) continue;
            updateSensorsAndControl(quad);
            const dReal *pos = dBodyGetPosition(quad->body);
            quad->fitness = pos[0];
            
            // Failure if 10 seconds have elapsed without sufficient x progress, or if hit.
            if ((simulationTime - quad->spawnTime >= 10.0 && (pos[0] - quad->spawnX < 2.0)) || quad->hit) {
                failedCount++;
            }
            
            // Also count if wheels are too close.
            bool wheelsTouch = false;
            for (int j = 0; j < 4; j++) {
                for (int k = j + 1; k < 4; k++) {
                    const dReal* posJ = dBodyGetPosition(quad->wheel[j]);
                    const dReal* posK = dBodyGetPosition(quad->wheel[k]);
                    double dx = posJ[0] - posK[0];
                    double dy = posJ[1] - posK[1];
                    double dz = posJ[2] - posK[2];
                    double dist = sqrt(dx * dx + dy * dy + dz * dz);
                    if (dist < (WHEEL_RADIUS * 2 * 1.1)) {
                        wheelsTouch = true;
                        break;
                    }
                }
                if (wheelsTouch)
                    break;
            }
            if (wheelsTouch)
                failedCount++;
        }
        
        // Check reset condition only once per second.
        if (simulationTime - lastResetCheck >= 1.0) {
            if (failedCount >= NUM_QUADRUPEDS / 2) {
                for (int i = 0; i < NUM_QUADRUPEDS; i++) {
                    replaceQuadruped(i);
                }
            }
            lastResetCheck = simulationTime;
        }
        
        dSpaceCollide(space, 0, &nearCallback);
        dWorldQuickStep(world, SIMULATION_DT);
        dJointGroupEmpty(contactGroup);
    }

    // Drawing code:
#if USE_TEXTURE
    dsSetTexture(DS_CHECKERED);
#endif
    dsSetColor(1.0, 1.0, 1.0);
    float groundSize[3] = {10.0f, 10.0f, 0.01f};
    float groundPos[3]  = {0.0f, 0.0f, 0.0f};
    float groundRot[12] = {1,0,0, 0,1,0, 0,0,1, 0,0,0};
    dsDrawBox(groundPos, groundRot, groundSize);
    dsSetTexture(DS_GROUND);

    for (int i = 0; i < 4; i++) {
        const dReal* pos = dGeomGetPosition(boundaryBoxes[i]);
        const dReal* rot = dGeomGetRotation(boundaryBoxes[i]);
        dVector3 dims;
        if (i < 2) {
            dims[0] = (40.0*2 + 1.0);
            dims[1] = 1.0;
            dims[2] = 5.0;
        } else {
            dims[0] = 1.0;
            dims[1] = (40.0*2);
            dims[2] = 5.0;
        }
        if (i == 3)
            dsSetTexture(DS_SKY);
        else
            dsSetTexture(DS_GROUND);
        dsDrawBox(pos, rot, dims);
    }

    drawObstacles();

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
        dVector3 right   = { bodyR[1], bodyR[5], bodyR[9] };
        dVector3 up      = { bodyR[2], bodyR[6], bodyR[10] };
        for (int i = 0; i < SENSOR_RAY_GRID; i++) {
            for (int j = 0; j < SENSOR_RAY_GRID; j++) {
                int idx = i * SENSOR_RAY_GRID + j;
                double horiz = ((double)j - (SENSOR_RAY_GRID - 1) / 2.0) * (SENSOR_CONE_ANGLE / (SENSOR_RAY_GRID - 1));
                double vert  = ((double)i - (SENSOR_RAY_GRID - 1) / 2.0) * (SENSOR_CONE_ANGLE / (SENSOR_RAY_GRID - 1));
                double cosVert = cos(vert);
                double sinVert = sin(vert);
                double cosHoriz = cos(horiz);
                double sinHoriz = sin(horiz);
                dVector3 rayDir;
                rayDir[0] = cosVert * cosHoriz * forward[0] + cosVert * sinHoriz * right[0] + sinVert * up[0];
                rayDir[1] = cosVert * cosHoriz * forward[1] + cosVert * sinHoriz * right[1] + sinVert * up[1];
                rayDir[2] = cosVert * cosHoriz * forward[2] + cosVert * sinHoriz * right[2] + sinVert * up[2];
                double sensorX = bodyPos[0] + forward[0]*(BODY_LENGTH*0.5);
                double sensorY = bodyPos[1] + forward[1]*(BODY_LENGTH*0.5);
                double sensorZ = bodyPos[2];
                double reading = quad->sensorValues[idx];
                double drawLength = DRAWN_MAX_LENGTH * reading;
                float rColor = (float)(1.0 - reading);
                float bColor = (float)(reading);
                glColor3f(rColor, 0.0f, bColor);
                glVertex3d(sensorX, sensorY, sensorZ);
                glVertex3d(sensorX + rayDir[0]*drawLength,
                           sensorY + rayDir[1]*drawLength,
                           sensorZ + rayDir[2]*drawLength);
            }
        }
    }
    glEnd();
    glPopAttrib();

    for (int i = 0; i < NUM_QUADRUPEDS; i++) {
        Quadruped *quad = &wobjects.quads[i];
        if (quad->body) {
            const dReal *bodyPos = dBodyGetPosition(quad->body);
            dsSetTexture(DS_WOOD);
            dVector3 sides = {BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT};
            dsDrawBox(bodyPos, dBodyGetRotation(quad->body), sides);
            for (int j = 0; j < 4; j++) {
                if (quad->leg[j]) {
                    dVector3 legSides = {LEG_WIDTH, LEG_WIDTH, LEG_LENGTH};
                    dsDrawBox(dBodyGetPosition(quad->leg[j]),
                              dBodyGetRotation(quad->leg[j]),
                              legSides);
                }
                if (quad->wheel[j]) {
                    dsDrawCylinder(dBodyGetPosition(quad->wheel[j]),
                                   dBodyGetRotation(quad->wheel[j]),
                                   WHEEL_RADIUS,
                                   WHEEL_WIDTH);
                }
            }
        }
    }
    char textBuffer[64];
    int countInside = 0;
    for (int i = 0; i < NUM_QUADRUPEDS; i++) {
        Quadruped *quad = &wobjects.quads[i];
        if (quad->body) {
            const dReal* pos = dBodyGetPosition(quad->body);
            if (fabs(pos[0]) <= 40.0 && fabs(pos[1]) <= 40.0)
                countInside++;
        }
    }
    sprintf(textBuffer, "Quadrupeds in Course: %d", countInside);
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    drawText(textBuffer, 10.0f, 10.0f);
    glPopAttrib();
}

/******************************************************************************
 * Drawstuff: Called once at the start.
 ******************************************************************************/
static void start()
{
    float xyz[3] = {3.0f, 3.0f, 1.5f};
    float hpr[3] = {130.0f, -25.0f, 0.0f};
    dsSetViewpoint(xyz, hpr);
    printf("Press 'p' to pause, 'q' to quit.\n");
}

/******************************************************************************
 * Drawstuff: Handle keyboard commands.
 ******************************************************************************/
static void command(int cmd)
{
    if (cmd == 'q')
        dsStop();
}

/******************************************************************************
 * Drawstuff: Called after the simulation ends.
 ******************************************************************************/
static void stop()
{
    // Nothing special.
}

/******************************************************************************
 * Helper for randomness.
 ******************************************************************************/
static double randomUniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

/******************************************************************************
 * Initialize the shared neural network with random parameters.
 ******************************************************************************/
static void initNetwork() {
    for (int i = 0; i < NN_HIDDEN; i++) {
        global_nn_b1[i] = randomUniform(-1.0, 1.0);
        for (int j = 0; j < NUM_RAYS; j++) {
            global_nn_W1[i][j] = randomUniform(-1.0, 1.0);
        }
    }
    for (int i = 0; i < NN_HIDDEN; i++) {
        global_nn_b2[i] = randomUniform(-1.0, 1.0);
        for (int j = 0; j < NN_HIDDEN; j++) {
            global_nn_W2[i][j] = randomUniform(-1.0, 1.0);
        }
    }
    for (int i = 0; i < NN_HIDDEN; i++) {
        global_nn_b3[i] = randomUniform(-1.0, 1.0);
        for (int j = 0; j < NN_HIDDEN; j++) {
            global_nn_W3[i][j] = randomUniform(-1.0, 1.0);
        }
    }
    for (int k = 0; k < NN_OUTPUTS; k++) {
        global_nn_b4[k] = randomUniform(-0.5, 0.5);
        for (int i = 0; i < NN_HIDDEN; i++) {
            global_nn_W4[k][i] = randomUniform(-1.0, 1.0);
        }
    }
}

/******************************************************************************
 * Main: sets up the ODE world, obstacles, network, and spawns quadrupeds.
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
    dsFunctions dsFunc = {
        DS_VERSION,
        &start,
        &simLoop,
        &command,
        &stop,
        "/Users/willnorden/Downloads/ode-0.16.6/drawstuff/textures"  
    };
#endif
    initODE();
    createStaticObstacles(5000, 1.0, 40.0);
    initNetwork();
    double margin = 2.0;
    double startX = -40.0 + margin;
    double startY = -25.0 + margin;
    double endY = 25.0 - margin;
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

/******************************************************************************
Compile:
g++ -std=c++11 -I/usr/local/include -L/usr/local/lib -o multiSim multiSim.cpp -lode -ldrawstuff -lm -framework GLUT -framework OpenGL -fopenmp

Run:
./multiSim
******************************************************************************/ 


/******************************************************************************
Challenges:

 * Configuring drawstuff

After installing the Open Dynamics Engine (ODE), I had to manually configure drawstuff to be included as a header file and 
then map those textures in my program to the parent working directory that the textures are stores in on my mac.

 * Quadrupeds breaking apart and flying everywhere

Occured when error stacked up too high somewhere, either from forces built up too high or the dt timesteps were too sparse. 
If this happened, the physics engine running Open Dynamics Engine woudln't know how to correct the error and things would 
just break and fly around. The replaceQuadruped function would try to place the quadrupeds back within the obstacle course, 
and so the simulation would turn into a massive storm of wheels and legs.

 * "Ghosts" of destroyed quadrupeds interfering with movement

When a quadruped is replaced, we must destroy both the body and its associated geom. Without doing this, the orphaned geoms
(still present in the space) may later be used in collision detection, causing a call to dBodyGetData() on an invalid 
(or already destroyed) body. This wasn't being done properly, and the remains of invisible, already destroyed 
quadrupeds were being detected in collisions.

 * Rays being treated like regular geometry caused quadrupeds to fly into the sky

In the collision callback we check if either geom is a ray (using dGeomGetClass) and immediately return so that sensor 
rays never contribute to physical contacts. This stops the quadrupeds from flying into the sky when their sensor rays 
hit obstacles.

 * Unexplained forward leaning behavior

 There was a situation where I shrank the width of the obstacle course as a quick fix for making sure there was no 
 way to get around the obstacles. I neglected to change the width of the locations that the quadrupeds could spawn
 into. Therefore, there were a small number (usually just 2) of quadrupeds being spawned outside of the walls of the 
 obstacle course with nothing blocking them. Those quadrupeds found that leaning forward would engage the wheels more
 somehow and would let them go faster towards the target wall. 
 
 The neural network learns from all of the quadrupeds and is shared between them to control hip angles. I implemented this 
 "Shared Brain" approach mainly so as to not slow down the simulation greatly. Therefore, the "outside" spawns were 
 rewarding the network for leaning forward, so it influenced all of the quadrupeds to lean forward in that way. In fact, 
 this behavior is horrible for navigating the obstacle course because the front wheels can never climb over anything. The 
 quick fix was to make sure all quadrupeds spawn inside the walls.

  * ODE INTERNAL ERROR 1: assertion "!bSafeNormalize4Fault" failed in dxNormalize4() [./odemath.h:53]

  This error would pop up after a few minutes of running and kill the program, which, of course, was a huge problem.
  Upon doing some research, it seems that this error indicates that ODE’s internal quaternion‐normalization routine 
  (dxNormalize4) is receiving a “bad” (nearly zero or otherwise degenerate) quaternion. In other words, one of the 
  quadruped parts (often when you pan/tilt the camera or after a while when physics become unstable) ends up with an 
  invalid rotation representation, and ODE aborts with the assertion failure. Implementing additional helper functions 
  to check and reset extreme rotations after each simulation step (in environment.cpp), as well as clamping of neural 
  network weights and biases (in neural_network.cpp) helped prevent the internal ODE error caused by degenerate rotation 
  matrices and possibly numerical instability in the learning updates, and I haven't recieved an issue of that kind since.

   * constant rewards causing poor performance

   A while after the simulation was actually at a point where I was happy with it, I realized the actually, the network 
   wasn't really learning at all. I watched it for a long time, and it never showed any improvement. In fact, it would
   get "stuck" in certain behaviors and never learn its way out of them, and wouldn't adapt to the environment. After a 
   while of this, I decided that I needed to know more about how the network was actually being trained. I already knew
   how to color objects from when I colored th green "target" ball, so I decided I wanted to use color to get direct
   visual feedback. I wrote some code that make quarupeds flash green or red based on their independent reward feedback,
   and blue if the global network recieve ssignificant updates to its weights. It was awesome to see blue flashes occur 
   then observe noticably different behavior from the quadrupeds. However, I noticed a large bug. Green flashes were 
   constant, and there were no red flashes, even if the quadrupeds were doing something awful like moving backwards or
   sideways. This was, of course, detrimental to their learning process and the reason was that I had previously 
   implemented rewarads and punishments in a non-ideal way. I had chosen to give a punishment if a quadruped had fallen,
   if its body or legs had collided with an object, or if it wasn't advancing towards the target. This is fine, however
   I had chosen to give rewards if any of these were NOT the case. That meant that I was constantly handing out rewards
   at every few time-steps just because they hadn't fallen or crashed into anything, meaning that the direction they were
   going was rather irrlevant to them as they were actually getting rewarded more than they were getting punished even
   while misbehaving. But it was the color flash feature that really helped me pinpoint this issue and focus on making
   my learning algorithm better using direct visual feedback.
 ******************************************************************************/