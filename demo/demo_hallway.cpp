// demo_hallway.cpp
// Simplified 1D hallway demo to verify network learning.
// Each agent navigates a separate hallway toward a target ball.

#define GL_SILENCE_DEPRECATION

#include "neural_network.h"
#include "physics.h"
#include "rendering.h"
#include "config.h"
#include "common.h"

#include <ode/ode.h>
#include <drawstuff/drawstuff.h>
#include <GLUT/glut.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// === Demo Configuration ===
namespace demo {
    constexpr int    NUM_AGENTS       = 5;
    constexpr double HALLWAY_LENGTH   = 20.0;
    constexpr double HALLWAY_WIDTH    = 4.0;
    constexpr double HALLWAY_SPACING  = 6.0;
    constexpr double GOAL_DISTANCE    = 1.5;
    constexpr double EPISODE_TIMEOUT  = 10.0;
    constexpr double COLLISION_PENALTY = -30.0;
    constexpr double TIMEOUT_PENALTY  = -50.0;
    constexpr double GOAL_REWARD      = 50.0;
}

// === Demo World State ===
struct DemoWorld {
    Quadruped agents[demo::NUM_AGENTS];
    dBodyID targets[demo::NUM_AGENTS];
    dGeomID targetGeoms[demo::NUM_AGENTS];
    dGeomID walls[demo::NUM_AGENTS * 4];  // 4 walls per hallway
    bool wallCollision[demo::NUM_AGENTS];
};

static DemoWorld demoWorld;
static dWorldID world;
static dSpaceID space;
static dJointGroupID contactGroup;
static dGeomID groundPlane;
static double simulationTime = 0.0;

// === Forward Declarations ===
static void createAgent(Quadruped* quad, double x, double y);
static void resetEpisode(int index, bool resetTarget);
static void nearCallback(void* data, dGeomID o1, dGeomID o2);
static void simLoop(int pause);
static void start();
static void command(int cmd);
static void stop();

// === Agent Creation ===
static void createAgent(Quadruped* quad, double x, double y) {
    using namespace config;
    
    double anchorX[4], anchorY[4];
    for (int i = 0; i < 4; i++) {
        double signX = (i < 2) ? -1.0 : 1.0;
        double signY = ((i % 2) == 0) ? -1.0 : 1.0;
        anchorX[i] = x + signX * (BODY_LENGTH * 0.5);
        anchorY[i] = y + signY * (BODY_WIDTH * 0.5);
    }
    
    double hipZ = LEG_LENGTH + 0.15;
    double torsoZ = hipZ + (BODY_HEIGHT * 0.5);
    
    // Create body
    quad->body = physics::createBox(world, space, BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT, BODY_MASS, &quad->bodyGeom);
    dBodySetPosition(quad->body, x, y, torsoZ);
    dBodySetData(quad->body, quad);
    
    // Initialize state
    quad->spawnX = x;
    quad->spawnTime = simulationTime;
    quad->prevX = x;
    quad->prevY = y;
    quad->blinkType = BLINK_NONE;
    quad->blinkCount = 0;
    quad->distanceToTarget = 100.0;
    quad->prevTargetDistance = 100.0;
    quad->prevCriticValue = 0.0;
    quad->collisionPenalty = 0.0;
    quad->ballVisible = false;
    quad->ballVisionValue = 1.0;
    quad->ballRayIndex = -1;
    quad->x = x;
    quad->y = y;
    quad->orientation = 0.0;
    
    // Create sensors
    for (int i = 0; i < NUM_RAYS; i++) {
        quad->raySensors[i] = dCreateRay(space, SENSOR_MAX_DIST);
        dGeomRaySet(quad->raySensors[i], x, y, torsoZ, 1, 0, 0);
        quad->sensorValues[i] = 1.0;
    }
    
    // Create legs and wheels
    for (int i = 0; i < 4; i++) {
        quad->leg[i] = physics::createBox(world, space, LEG_WIDTH, LEG_WIDTH, LEG_LENGTH, LEG_MASS, &quad->legGeom[i]);
        dBodySetPosition(quad->leg[i], anchorX[i], anchorY[i], hipZ - LEG_LENGTH * 0.5);
        dBodySetData(quad->leg[i], quad);
        
        quad->hip[i] = dJointCreateHinge(world, nullptr);
        dJointAttach(quad->hip[i], quad->body, quad->leg[i]);
        dJointSetHingeAnchor(quad->hip[i], anchorX[i], anchorY[i], hipZ);
        dJointSetHingeAxis(quad->hip[i], 0, 1, 0);
        dJointSetHingeParam(quad->hip[i], dParamFMax, HIP_FMAX);
        
        // Lock hips for 1D movement
        dJointSetHingeParam(quad->hip[i], dParamLoStop, 0.0);
        dJointSetHingeParam(quad->hip[i], dParamHiStop, 0.0);
        
        quad->wheel[i] = physics::createCylinder(world, space, WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS, true, &quad->wheelTransform[i]);
        double offsetY = ((i % 2) == 0) ? -(LEG_WIDTH / 2.0 + WHEEL_RADIUS) : (LEG_WIDTH / 2.0 + WHEEL_RADIUS);
        dBodySetPosition(quad->wheel[i], anchorX[i], anchorY[i] + offsetY, hipZ - LEG_LENGTH);
        dBodySetData(quad->wheel[i], quad);
        
        quad->wheelJoint[i] = dJointCreateHinge(world, nullptr);
        dJointAttach(quad->wheelJoint[i], quad->leg[i], quad->wheel[i]);
        dJointSetHingeAnchor(quad->wheelJoint[i], anchorX[i], anchorY[i] + offsetY, hipZ - LEG_LENGTH);
        dJointSetHingeAxis(quad->wheelJoint[i], 0, 1, 0);
        dJointSetHingeParam(quad->wheelJoint[i], dParamFMax, WHEEL_FMAX);
    }
}

static void resetEpisode(int index, bool resetTarget) {
    using namespace demo;
    
    Quadruped* quad = &demoWorld.agents[index];
    demoWorld.wallCollision[index] = false;
    quad->ballVisible = false;
    quad->ballVisionValue = 1.0;
    
    // Destroy old agent
    physics::destroyQuadruped(quad);
    
    // Respawn at hallway center
    double yPos = index * HALLWAY_SPACING;
    createAgent(quad, 0.0, yPos);
    
    // Optionally reset target
    if (resetTarget) {
        double targetX = randomUniform(-8.0, 8.0);
        if (std::fabs(targetX) < 2.0) {
            targetX = (targetX > 0) ? 3.0 : -3.0;
        }
        dBodySetPosition(demoWorld.targets[index], targetX, yPos, 0.5);
        dBodySetLinearVel(demoWorld.targets[index], 0, 0, 0);
        dBodySetAngularVel(demoWorld.targets[index], 0, 0, 0);
        quad->targetX = targetX;
        quad->targetY = yPos;
    } else {
        const dReal* tPos = dBodyGetPosition(demoWorld.targets[index]);
        quad->targetX = tPos[0];
        quad->targetY = tPos[1];
    }
    
    // Reset state
    quad->prevCriticValue = 0.0;
    const dReal* pos = dBodyGetPosition(quad->body);
    double dx = quad->targetX - pos[0];
    double dy = quad->targetY - pos[1];
    quad->prevTargetDistance = std::sqrt(dx * dx + dy * dy);
    quad->spawnTime = simulationTime;
}

static void nearCallback(void* /*data*/, dGeomID o1, dGeomID o2) {
    if (dGeomGetClass(o1) == dRayClass || dGeomGetClass(o2) == dRayClass) return;
    
    dContact contacts[4];
    int n = dCollide(o1, o2, 4, &contacts[0].geom, sizeof(dContact));
    
    for (int i = 0; i < n; i++) {
        contacts[i].surface.mode = dContactBounce | dContactSoftERP | dContactSoftCFM;
        contacts[i].surface.mu = 500.0;
        contacts[i].surface.bounce = 0.05;
        contacts[i].surface.soft_erp = 0.8;
        contacts[i].surface.soft_cfm = 1e-5;
        
        dJointID c = dJointCreateContact(world, contactGroup, &contacts[i]);
        dJointAttach(c, dGeomGetBody(o1), dGeomGetBody(o2));
        
        if (o1 == groundPlane || o2 == groundPlane) continue;
        
        // Track wall collisions
        bool isWallCollision = false;
        for (int w = 0; w < demo::NUM_AGENTS * 4; w++) {
            if (o1 == demoWorld.walls[w] || o2 == demoWorld.walls[w]) {
                isWallCollision = true;
                break;
            }
        }
        
        if (isWallCollision) {
            dBodyID b1 = dGeomGetBody(o1);
            dBodyID b2 = dGeomGetBody(o2);
            
            for (int agentIdx = 0; agentIdx < demo::NUM_AGENTS; agentIdx++) {
                Quadruped* agent = &demoWorld.agents[agentIdx];
                if (agent->body == b1 || agent->body == b2 ||
                    agent->leg[0] == b1 || agent->leg[0] == b2 ||
                    agent->leg[1] == b1 || agent->leg[1] == b2 ||
                    agent->leg[2] == b1 || agent->leg[2] == b2 ||
                    agent->leg[3] == b1 || agent->leg[3] == b2 ||
                    agent->wheel[0] == b1 || agent->wheel[0] == b2 ||
                    agent->wheel[1] == b1 || agent->wheel[1] == b2 ||
                    agent->wheel[2] == b1 || agent->wheel[2] == b2 ||
                    agent->wheel[3] == b1 || agent->wheel[3] == b2) {
                    demoWorld.wallCollision[agentIdx] = true;
                    break;
                }
            }
        }
    }
}

static void simLoop(int pause) {
    using namespace config;
    using namespace demo;
    
    if (!pause) {
        simulationTime += SIMULATION_DT;
        
        for (int i = 0; i < NUM_AGENTS; i++) {
            Quadruped* quad = &demoWorld.agents[i];
            if (!quad->body) continue;
            
            // Update state
            const dReal* pos = dBodyGetPosition(quad->body);
            const dReal* rot = dBodyGetRotation(quad->body);
            quad->x = pos[0];
            quad->y = pos[1];
            quad->orientation = std::atan2(rot[4], rot[0]);
            
            // Update sensors
            bool ballHit = false;
            double bestBallVal = 1.0;
            int bestBallIdx = -1;
            
            for (int r = 0; r < NUM_RAYS; r++) {
                double angle = quad->orientation + (2 * M_PI * r / NUM_RAYS);
                dGeomRaySet(quad->raySensors[r], pos[0], pos[1], pos[2], 
                           std::cos(angle), std::sin(angle), 0.0);
                
                dContactGeom contact;
                if (dCollide(quad->raySensors[r], groundPlane, 1, &contact, sizeof(dContactGeom))) {
                    quad->sensorValues[r] = contact.depth / SENSOR_MAX_DIST;
                } else {
                    quad->sensorValues[r] = 1.0;
                }
                
                // Check walls
                for (int w = 0; w < NUM_AGENTS * 4; w++) {
                    if (dCollide(quad->raySensors[r], demoWorld.walls[w], 1, &contact, sizeof(dContactGeom))) {
                        double d = contact.depth / SENSOR_MAX_DIST;
                        if (d < quad->sensorValues[r]) quad->sensorValues[r] = d;
                    }
                }
                
                // Check target
                if (dCollide(quad->raySensors[r], demoWorld.targetGeoms[i], 1, &contact, sizeof(dContactGeom))) {
                    double d = contact.depth / SENSOR_MAX_DIST;
                    if (d < quad->sensorValues[r]) quad->sensorValues[r] = d;
                    if (d < bestBallVal) {
                        bestBallVal = d;
                        ballHit = true;
                        bestBallIdx = r;
                    }
                }
            }
            
            quad->ballVisible = ballHit;
            quad->ballVisionValue = ballHit ? bestBallVal : 1.0;
            quad->ballRayIndex = bestBallIdx;
            
            // Compute reward
            const dReal* tPos = dBodyGetPosition(demoWorld.targets[i]);
            quad->targetX = tPos[0];
            quad->targetY = tPos[1];
            
            double dx = tPos[0] - pos[0];
            double dy = tPos[1] - pos[1];
            double dist = std::sqrt(dx * dx + dy * dy);
            quad->distanceToTarget = dist;
            
            double reward = 0.0;
            
            // Goal reached
            if (dist < GOAL_DISTANCE) {
                reward = GOAL_REWARD;
                double actions[ACTOR_OUTPUTS];
                runNeuralNetwork(quad, reward, actions);
                resetEpisode(i, true);
                continue;
            }
            
            // Wall collision
            if (demoWorld.wallCollision[i]) {
                reward = COLLISION_PENALTY;
                double actions[ACTOR_OUTPUTS];
                runNeuralNetwork(quad, reward, actions);
                resetEpisode(i, false);
                continue;
            }
            
            // Timeout
            if (simulationTime - quad->spawnTime > EPISODE_TIMEOUT) {
                double distancePenalty = dist * 5.0;
                reward = TIMEOUT_PENALTY - distancePenalty;
                double actions[ACTOR_OUTPUTS];
                runNeuralNetwork(quad, reward, actions);
                resetEpisode(i, false);
                continue;
            }
            
            // Shaping reward
            reward += 10.0 * (quad->prevTargetDistance - dist);
            quad->prevTargetDistance = dist;
            reward -= 0.01;  // Time penalty
            
            // Run network
            double actions[ACTOR_OUTPUTS];
            int action = runNeuralNetwork(quad, reward, actions);
            
            // Execute action (1D movement only)
            double wheelVel = 0;
            if (action == 0) wheelVel = -20;      // Backward
            else if (action == 3) wheelVel = 20;  // Forward
            
            for (int w = 0; w < 4; w++) {
                dJointSetHingeParam(quad->wheelJoint[w], dParamVel, wheelVel);
            }
        }
        
        applyBatchUpdate();
        
        dSpaceCollide(space, nullptr, &nearCallback);
        dWorldQuickStep(world, SIMULATION_DT);
        dJointGroupEmpty(contactGroup);
    }
    
    // === Rendering ===
    dsSetColor(1.0, 1.0, 1.0);
    
    // Draw walls
    for (int i = 0; i < NUM_AGENTS * 4; i++) {
        const dReal* pos = dGeomGetPosition(demoWorld.walls[i]);
        int wallIndex = i % 4;
        dVector3 size;
        if (wallIndex < 2) {
            size[0] = HALLWAY_LENGTH; size[1] = 0.2; size[2] = 1.0;
        } else {
            size[0] = 0.2; size[1] = HALLWAY_WIDTH; size[2] = 1.0;
        }
        dsDrawBox(pos, dGeomGetRotation(demoWorld.walls[i]), size);
    }
    
    // Draw sensor rays
    rendering::drawAllSensorRays(demoWorld.agents, NUM_AGENTS);
    
    // Draw agents and targets
    for (int i = 0; i < NUM_AGENTS; i++) {
        Quadruped* q = &demoWorld.agents[i];
        if (q->body) {
            dsSetTexture(DS_WOOD);
            
            // Highlight if seeing ball
            if (q->ballVisible) {
                double phase = std::fmod(simulationTime, 0.4);
                if (phase < 0.2) {
                    dsSetColor(1.0, 1.0, 0.0);
                } else {
                    dsSetColor(1.0, 1.0, 1.0);
                }
            } else {
                dsSetColor(1.0, 1.0, 1.0);
            }
            
            dVector3 sides = {BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT};
            dsDrawBox(dBodyGetPosition(q->body), dBodyGetRotation(q->body), sides);
            
            dsSetColor(0.8, 0.8, 0.0);
            for (int j = 0; j < 4; j++) {
                dVector3 lSides = {LEG_WIDTH, LEG_WIDTH, LEG_LENGTH};
                dsDrawBox(dBodyGetPosition(q->leg[j]), dBodyGetRotation(q->leg[j]), lSides);
                
                dsSetColor(0.2, 0.2, 0.2);
                dsDrawCylinder(dBodyGetPosition(q->wheel[j]), dBodyGetRotation(q->wheel[j]), 
                              WHEEL_RADIUS, WHEEL_WIDTH);
            }
            
            // Draw target
            dsSetColor(0.0, 1.0, 0.0);
            const dReal* tPos = dBodyGetPosition(demoWorld.targets[i]);
            dsDrawSphere(tPos, dBodyGetRotation(demoWorld.targets[i]), 0.5);
        }
    }
}

static void start() {
    float xyz[3] = {0.0f, -10.0f, 20.0f};
    float hpr[3] = {90.0f, -45.0f, 0.0f};
    dsSetViewpoint(xyz, hpr);
    
    std::printf("Demo Controls:\n");
    std::printf("  'p' - Pause\n");
    std::printf("  'q' - Quit\n");
    std::printf("  '+' - Zoom in\n");
    std::printf("  '-' - Zoom out\n");
}

static void command(int cmd) {
    if (cmd == 'q') {
        dsStop();
        return;
    }
    
    if (cmd == '+' || cmd == '=' || cmd == '-' || cmd == '_') {
        float xyz[3], hpr[3];
        dsGetViewpoint(xyz, hpr);
        if (cmd == '+' || cmd == '=') {
            xyz[2] *= 0.9f;
        } else {
            xyz[2] *= 1.1f;
        }
        dsSetViewpoint(xyz, hpr);
    }
}

static void stop() {}

static void setupDemo() {
    using namespace demo;
    
    dInitODE();
    world = dWorldCreate();
    space = dHashSpaceCreate(nullptr);
    contactGroup = dJointGroupCreate(0);
    dWorldSetGravity(world, 0, 0, -9.81);
    groundPlane = dCreatePlane(space, 0, 0, 1, 0);
    
    initNetwork();
    
    for (int i = 0; i < NUM_AGENTS; i++) {
        double yPos = i * HALLWAY_SPACING;
        
        demoWorld.wallCollision[i] = false;
        
        // Side walls
        demoWorld.walls[i * 4 + 0] = dCreateBox(space, HALLWAY_LENGTH, 0.2, 1.0);
        dGeomSetPosition(demoWorld.walls[i * 4 + 0], 0, yPos - HALLWAY_WIDTH / 2.0, 0.5);
        
        demoWorld.walls[i * 4 + 1] = dCreateBox(space, HALLWAY_LENGTH, 0.2, 1.0);
        dGeomSetPosition(demoWorld.walls[i * 4 + 1], 0, yPos + HALLWAY_WIDTH / 2.0, 0.5);
        
        // End walls
        demoWorld.walls[i * 4 + 2] = dCreateBox(space, 0.2, HALLWAY_WIDTH, 1.0);
        dGeomSetPosition(demoWorld.walls[i * 4 + 2], -HALLWAY_LENGTH / 2.0, yPos, 0.5);
        
        demoWorld.walls[i * 4 + 3] = dCreateBox(space, 0.2, HALLWAY_WIDTH, 1.0);
        dGeomSetPosition(demoWorld.walls[i * 4 + 3], HALLWAY_LENGTH / 2.0, yPos, 0.5);
        
        // Target
        demoWorld.targets[i] = dBodyCreate(world);
        dMass m;
        dMassSetSphere(&m, 1.0, 0.5);
        dBodySetMass(demoWorld.targets[i], &m);
        demoWorld.targetGeoms[i] = dCreateSphere(space, 0.5);
        dGeomSetBody(demoWorld.targetGeoms[i], demoWorld.targets[i]);
        
        resetEpisode(i, true);
    }
}

int main(int argc, char** argv) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    dsFunctions fn;
    fn.version = DS_VERSION;
    fn.start = &start;
    fn.step = &simLoop;
    fn.command = &command;
    fn.stop = &stop;
    fn.path_to_textures = TEXTURE_PATH;
    
    setupDemo();
    
    int dummyArgc = 1;
    char* dummyArgv[] = {const_cast<char*>("demo_hallway"), nullptr};
    if (argc == 0) {
        argc = dummyArgc;
        argv = dummyArgv;
    }
    
    dsSimulationLoop(argc, argv, 800, 600, &fn);
    
    dCloseODE();
    return 0;
}
