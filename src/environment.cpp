// environment.cpp
// Main simulation environment using ODE physics and A2C reinforcement learning.

#define GL_SILENCE_DEPRECATION

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "environment.h"
#include "neural_network.h"
#include "physics.h"
#include "rendering.h"
#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// Static instance pointer for callbacks
Environment* Environment::instance_ = nullptr;

// External neural network state
extern double globalWeightChangeBlink;
extern bool globalInitNetworkCalled;

Environment::Environment() 
    : world_(nullptr)
    , space_(nullptr)
    , contactGroup_(nullptr)
    , groundPlane_(nullptr)
    , targetBall_(nullptr)
    , targetBallGeom_(nullptr)
    , numObstacles_(0)
    , numQuadrupeds_(0)
    , simulationTime_(0.0) {
    
    for (int i = 0; i < 4; i++) {
        boundaryBoxes_[i] = nullptr;
    }
    instance_ = this;
}

Environment::~Environment() {
    cleanup();
}

void Environment::initialize() {
    physics::initWorld(world_, space_, contactGroup_, groundPlane_);
    physics::createBoundaryWalls(space_, boundaryBoxes_);
    physics::createTargetBall(world_, space_, targetBall_, targetBallGeom_);
}

void Environment::createObstacles(int count) {
    using namespace config;
    
    numObstacles_ = 0;
    double innerMargin = 1.0;
    double outerMargin = 1.0;
    
    for (int i = 0; i < count && numObstacles_ < MAX_OBSTACLES; i++) {
        double x = -60.0 + innerMargin + (static_cast<double>(rand()) / RAND_MAX) * 
                   ((60.0 - outerMargin) - (-60.0 + innerMargin));
        double y = WORLD_Y_MIN + innerMargin + (static_cast<double>(rand()) / RAND_MAX) * 
                   ((WORLD_Y_MAX - outerMargin) - (WORLD_Y_MIN + innerMargin));
        
        double sx = 0.5 + (static_cast<double>(rand()) / RAND_MAX) * 1.0;
        double sy = 0.5 + (static_cast<double>(rand()) / RAND_MAX) * 1.0;
        double sz = 0.1 + (static_cast<double>(rand()) / RAND_MAX) * (2.0 - 0.1);
        double pz = sz / 2.0;
        
        dGeomID geom = dCreateBox(space_, sx, sy, sz);
        dGeomSetPosition(geom, x, y, pz);
        dGeomSetData(geom, const_cast<char*>("obstacle"));
        
        obstacles_[numObstacles_] = geom;
        obstacleSizes_[numObstacles_][0] = sx;
        obstacleSizes_[numObstacles_][1] = sy;
        obstacleSizes_[numObstacles_][2] = sz;
        numObstacles_++;
    }
}

void Environment::spawnAgents(int count) {
    using namespace config;
    
    numQuadrupeds_ = count;
    double margin = 2.0;
    double startX = WORLD_X_MIN + margin;
    double startY = WORLD_Y_MIN + margin;
    double endY = WORLD_Y_MAX - margin;
    double spacing = (count > 1) ? (endY - startY) / (count - 1) : 0;
    
    for (int i = 0; i < count; i++) {
        double x = startX;
        double y = startY + i * spacing;
        physics::createQuadruped(world_, space_, &quads_[i], x, y, 
                                 simulationTime_, targetBall_,
                                 obstacles_, obstacleSizes_, numObstacles_);
    }
}

dsFunctions Environment::getCallbacks() {
    dsFunctions fn;
    fn.version = DS_VERSION;
    fn.start = &Environment::startCallback;
    fn.step = &Environment::simLoopCallback;
    fn.command = &Environment::commandCallback;
    fn.stop = &Environment::stopCallback;
    fn.path_to_textures = TEXTURE_PATH;
    return fn;
}

void Environment::cleanup() {
    if (world_) {
        physics::cleanup(world_, space_, contactGroup_);
        world_ = nullptr;
        space_ = nullptr;
        contactGroup_ = nullptr;
    }
}

void Environment::startCallback() {
    if (!instance_) return;
    
    const dReal* quadPos = dBodyGetPosition(instance_->quads_[0].body);
    float xyz[3] = {
        static_cast<float>(quadPos[0] + 3.0),
        static_cast<float>(quadPos[1] - 2.0),
        static_cast<float>(quadPos[2] + 2.5)
    };
    float hpr[3] = {90.0f, -15.0f, 0.0f};
    dsSetViewpoint(xyz, hpr);
    
    std::printf("MultiSim Controls:\n");
    std::printf("  'p' - Pause simulation\n");
    std::printf("  'q' - Quit\n");
}

void Environment::commandCallback(int cmd) {
    if (cmd == 'q') {
        dsStop();
    }
}

void Environment::stopCallback() {
    // Nothing to do
}

void Environment::nearCallback(void* /*data*/, dGeomID o1, dGeomID o2) {
    using namespace config;
    
    if (!instance_) return;
    
    // Skip collisions involving rays
    if (dGeomGetClass(o1) == dRayClass || dGeomGetClass(o2) == dRayClass) {
        return;
    }
    
    dContact contacts[8];
    int n = dCollide(o1, o2, 8, &contacts[0].geom, sizeof(dContact));
    
    for (int i = 0; i < n; i++) {
        contacts[i].surface.mode = dContactBounce | dContactSoftERP | dContactSoftCFM;
        contacts[i].surface.mu = 1200.0;
        contacts[i].surface.bounce = 0.7;
        contacts[i].surface.soft_erp = 1.0;
        contacts[i].surface.soft_cfm = 1e-4;
        
        dJointID c = dJointCreateContact(instance_->world_, instance_->contactGroup_, &contacts[i]);
        dJointAttach(c, dGeomGetBody(o1), dGeomGetBody(o2));
        
        // Skip ground collisions for penalty calculation
        if (o1 == instance_->groundPlane_ || o2 == instance_->groundPlane_) {
            continue;
        }
        
        void* data1 = dGeomGetData(o1);
        void* data2 = dGeomGetData(o2);
        
        // Handle obstacle collisions
        if (data1 == static_cast<void*>(const_cast<char*>("obstacle"))) {
            dBodyID b = dGeomGetBody(o2);
            if (b) {
                Quadruped* quad = static_cast<Quadruped*>(dBodyGetData(b));
                if (quad) {
                    quad->collisionPenalty += COLLISION_PENALTY;
                }
            }
        } else if (data2 == static_cast<void*>(const_cast<char*>("obstacle"))) {
            dBodyID b = dGeomGetBody(o1);
            if (b) {
                Quadruped* quad = static_cast<Quadruped*>(dBodyGetData(b));
                if (quad) {
                    quad->collisionPenalty += COLLISION_PENALTY;
                }
            }
        }
        
        // Handle wall collisions
        for (int w = 0; w < 4; w++) {
            if (o1 == instance_->boundaryBoxes_[w] || o2 == instance_->boundaryBoxes_[w]) {
                dBodyID b = (o1 == instance_->boundaryBoxes_[w]) ? dGeomGetBody(o2) : dGeomGetBody(o1);
                if (b) {
                    Quadruped* quad = static_cast<Quadruped*>(dBodyGetData(b));
                    if (quad) {
                        quad->collisionPenalty += WALL_COLLISION_PENALTY;
                    }
                }
                break;
            }
        }
    }
}

bool Environment::hasFallen(Quadruped* quad) {
    const dReal* R = dBodyGetRotation(quad->body);
    double up_z = R[10];
    return (up_z < config::FALLING_THRESHOLD);
}

double Environment::getEnvironmentHeight(double x, double y) {
    double height = 0.0;
    
    for (int i = 0; i < numObstacles_; i++) {
        const dReal* pos = dGeomGetPosition(obstacles_[i]);
        double sx = obstacleSizes_[i][0];
        double sy = obstacleSizes_[i][1];
        double half_sx = sx / 2.0;
        double half_sy = sy / 2.0;
        
        if (x >= pos[0] - half_sx && x <= pos[0] + half_sx &&
            y >= pos[1] - half_sy && y <= pos[1] + half_sy) {
            double top = pos[2] + (obstacleSizes_[i][2] / 2.0);
            if (top > height) {
                height = top;
            }
        }
    }
    
    return height;
}

void Environment::replaceQuadruped(int index) {
    using namespace config;
    
    Quadruped* quad = &quads_[index];
    physics::destroyQuadruped(quad);
    
    quad->respawnCount++;
    quad->lastRespawnTime = simulationTime_;
    quad->fitness = 0;
    
    // Respawn at starting position
    double margin = 2.0;
    double startX = WORLD_X_MIN + margin;
    double startY = WORLD_Y_MIN + margin;
    double endY = WORLD_Y_MAX - margin;
    double spacing = (numQuadrupeds_ > 1) ? (endY - startY) / (numQuadrupeds_ - 1) : 0;
    double x = startX;
    double y = startY + index * spacing;
    
    physics::createQuadruped(world_, space_, quad, x, y,
                             simulationTime_, targetBall_,
                             obstacles_, obstacleSizes_, numObstacles_);
    
    quad->prevCriticValue = 0.0;
}

void Environment::updateSensorsAndControl(Quadruped* quad) {
    using namespace config;
    
    const dReal* bodyPos = dBodyGetPosition(quad->body);
    const dReal* bodyR = dBodyGetRotation(quad->body);
    
    // Update position and orientation
    quad->x = bodyPos[0];
    quad->y = bodyPos[1];
    quad->orientation = std::atan2(bodyR[4], bodyR[0]);
    
    // Update sensor rays (full circle)
    for (int i = 0; i < NUM_RAYS; i++) {
        double angle = quad->orientation + (2 * M_PI * i / NUM_RAYS);
        double sensorX = bodyPos[0];
        double sensorY = bodyPos[1];
        double sensorZ = bodyPos[2];
        double dirX = std::cos(angle);
        double dirY = std::sin(angle);
        double dirZ = 0.0;
        dGeomRaySet(quad->raySensors[i], sensorX, sensorY, sensorZ, dirX, dirY, dirZ);
    }
    
    // Sensor collision checks
    bool ballHit = false;
    double bestBallVal = 1.0;
    int bestBallIdx = -1;
    
    for (int idx = 0; idx < NUM_RAYS; idx++) {
        double minReading = 1.0;
        dContactGeom contact;
        
        // Check ground
        int n = dCollide(quad->raySensors[idx], groundPlane_, 1, &contact, sizeof(dContactGeom));
        if (n > 0) {
            double reading = contact.depth / SENSOR_MAX_DIST;
            if (reading < minReading) minReading = reading;
        }
        
        // Check obstacles
        for (int obs = 0; obs < numObstacles_; obs++) {
            n = dCollide(quad->raySensors[idx], obstacles_[obs], 1, &contact, sizeof(dContactGeom));
            if (n > 0) {
                double reading = contact.depth / SENSOR_MAX_DIST;
                if (reading < minReading) minReading = reading;
            }
        }
        
        // Check walls
        for (int w = 0; w < 4; w++) {
            n = dCollide(quad->raySensors[idx], boundaryBoxes_[w], 1, &contact, sizeof(dContactGeom));
            if (n > 0) {
                double reading = contact.depth / SENSOR_MAX_DIST;
                if (reading < minReading) minReading = reading;
            }
        }
        
        // Check target ball
        n = dCollide(quad->raySensors[idx], targetBallGeom_, 1, &contact, sizeof(dContactGeom));
        if (n > 0) {
            double reading = contact.depth / SENSOR_MAX_DIST;
            if (reading < minReading) minReading = reading;
            if (reading < bestBallVal) {
                bestBallVal = reading;
                ballHit = true;
                bestBallIdx = idx;
            }
        }
        
        if (minReading > 1.0) minReading = 1.0;
        quad->sensorValues[idx] = minReading;
    }
    
    quad->ballVisible = ballHit;
    quad->ballVisionValue = ballHit ? bestBallVal : 1.0;
    quad->ballRayIndex = bestBallIdx;
    
    // Compute target distance
    const dReal* targetPos = dBodyGetPosition(targetBall_);
    double dx = targetPos[0] - bodyPos[0];
    double dy = targetPos[1] - bodyPos[1];
    double dz = targetPos[2] - bodyPos[2];
    double currentTargetDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    quad->distanceToTarget = currentTargetDistance;
    quad->targetX = targetPos[0];
    quad->targetY = targetPos[1];
    
    // === REWARD CALCULATION ===
    double reward = 0.0;
    
    // Goal reached (terminal success)
    if (currentTargetDistance < GOAL_DISTANCE) {
        reward = GOAL_REWARD;
        double actions[ACTOR_OUTPUTS];
        runNeuralNetwork(quad, reward, actions);
        int index = static_cast<int>(quad - quads_);
        replaceQuadruped(index);
        return;
    }
    
    // Timeout (terminal failure)
    if (simulationTime_ - quad->spawnTime > EPISODE_TIMEOUT) {
        reward = TIMEOUT_PENALTY;
        double actions[ACTOR_OUTPUTS];
        runNeuralNetwork(quad, reward, actions);
        int index = static_cast<int>(quad - quads_);
        replaceQuadruped(index);
        return;
    }
    
    // Distance improvement reward (main signal)
    double distanceImprovement = quad->prevTargetDistance - currentTargetDistance;
    reward += DISTANCE_SCALE * distanceImprovement;
    quad->prevTargetDistance = currentTargetDistance;
    
    // Collision penalty
    if (quad->collisionPenalty < 0) {
        reward += quad->collisionPenalty * 0.3;
    }
    quad->collisionPenalty = 0.0;
    
    // Obstacle clearance penalty
    const dReal* bodyVel = dBodyGetLinearVel(quad->body);
    double speed = std::sqrt(bodyVel[0] * bodyVel[0] + bodyVel[1] * bodyVel[1]);
    
    if (speed > 0.3) {
        double minSensorDist = 1.0;
        for (int i = 0; i < NUM_RAYS; i++) {
            if (quad->sensorValues[i] < minSensorDist) {
                minSensorDist = quad->sensorValues[i];
            }
        }
        
        if (minSensorDist < 0.3) {
            reward -= 1.0;
        } else if (minSensorDist < 0.5) {
            reward -= 0.3;
        }
    }
    
    // Milestone bonuses
    double forwardProgress = bodyPos[0] - quad->prevX;
    if (forwardProgress > 3.0) {
        reward += 20.0;
    } else if (forwardProgress > 1.5) {
        reward += 10.0;
    }
    
    // Stability penalty
    double upright = bodyR[10];
    if (upright < 0.6) {
        reward -= 2.0;
    }
    
    // Time penalty
    reward += TIME_PENALTY;
    
    quad->prevX = bodyPos[0];
    quad->prevY = bodyPos[1];
    
    // Visual feedback
    if (quad->blinkCount == 0) {
        if (reward >= 3.0) {
            quad->blinkType = BLINK_GREEN;
            quad->blinkCount = 1;
            quad->blinkStartTime = simulationTime_;
        } else if (reward <= -1.5) {
            quad->blinkType = BLINK_RED;
            quad->blinkCount = 1;
            quad->blinkStartTime = simulationTime_;
        } else {
            quad->blinkType = BLINK_NONE;
        }
    }
    
    // Check if fallen
    if (hasFallen(quad)) {
        reward = FALLING_PENALTY;
    }
    
    // Run neural network
    double actions[ACTOR_OUTPUTS];
    int chosen_action = runNeuralNetwork(quad, reward, actions);
    
    // Execute action
    switch (chosen_action) {
        case 0:  // Forward
            for (int i = 0; i < 4; i++) {
                dJointSetHingeParam(quad->wheelJoint[i], dParamVel, WHEEL_VEL_FORWARD);
            }
            break;
        case 1:  // Turn left
            dJointSetHingeParam(quad->wheelJoint[0], dParamVel, WHEEL_VEL_TURN_FAST);
            dJointSetHingeParam(quad->wheelJoint[2], dParamVel, WHEEL_VEL_TURN_FAST);
            dJointSetHingeParam(quad->wheelJoint[1], dParamVel, WHEEL_VEL_TURN_SLOW);
            dJointSetHingeParam(quad->wheelJoint[3], dParamVel, WHEEL_VEL_TURN_SLOW);
            break;
        case 2:  // Turn right
            dJointSetHingeParam(quad->wheelJoint[0], dParamVel, -WHEEL_VEL_TURN_FAST);
            dJointSetHingeParam(quad->wheelJoint[2], dParamVel, -WHEEL_VEL_TURN_FAST);
            dJointSetHingeParam(quad->wheelJoint[1], dParamVel, -WHEEL_VEL_TURN_SLOW);
            dJointSetHingeParam(quad->wheelJoint[3], dParamVel, -WHEEL_VEL_TURN_SLOW);
            break;
        case 3:  // Backward
            for (int i = 0; i < 4; i++) {
                dJointSetHingeParam(quad->wheelJoint[i], dParamVel, WHEEL_VEL_BACK);
            }
            break;
    }
}

void Environment::simLoopCallback(int pause) {
    using namespace config;
    
    if (!instance_) return;
    
    if (!pause) {
        instance_->simulationTime_ += SIMULATION_DT;
        
        // Process all quadrupeds
        for (int i = 0; i < instance_->numQuadrupeds_; i++) {
            Quadruped* quad = &instance_->quads_[i];
            if (!quad->body) continue;
            
            instance_->updateSensorsAndControl(quad);
            
            const dReal* pos = dBodyGetPosition(quad->body);
            quad->fitness = pos[0];
            
            if (instance_->hasFallen(quad)) {
                instance_->replaceQuadruped(i);
                continue;
            }
        }
        
        // Batch update neural network
        applyBatchUpdate();
        
        // Global weight change blink
        if (globalWeightChangeBlink > 0) {
            for (int i = 0; i < instance_->numQuadrupeds_; i++) {
                instance_->quads_[i].blinkType = BLINK_BLUE;
                instance_->quads_[i].blinkCount = 2;
                instance_->quads_[i].blinkStartTime = instance_->simulationTime_;
            }
            globalWeightChangeBlink = 0.0;
        }
        
        // Physics step
        dSpaceCollide(instance_->space_, nullptr, &Environment::nearCallback);
        dWorldQuickStep(instance_->world_, SIMULATION_DT);
        dJointGroupEmpty(instance_->contactGroup_);
        
        // Normalize rotations
        for (int i = 0; i < instance_->numQuadrupeds_; i++) {
            Quadruped* quad = &instance_->quads_[i];
            if (quad->body) {
                physics::safeNormalizeBodyRotation(quad->body);
            }
        }
        physics::safeNormalizeBodyRotation(instance_->targetBall_);
    }
    
    // === RENDERING ===
    rendering::drawGround();
    rendering::drawBoundaryWalls(instance_->boundaryBoxes_);
    rendering::drawObstacles(instance_->obstacles_, instance_->obstacleSizes_, instance_->numObstacles_);
    rendering::drawObstacleLines(instance_->obstacles_, instance_->obstacleSizes_, instance_->numObstacles_);
    rendering::drawTargetBall(instance_->targetBall_);
    rendering::drawAllSensorRays(instance_->quads_, instance_->numQuadrupeds_);
    
    for (int i = 0; i < instance_->numQuadrupeds_; i++) {
        rendering::drawQuadruped(&instance_->quads_[i], instance_->simulationTime_);
    }
    
    // Draw HUD
    char textBuffer[64];
    int countInside = 0;
    for (int i = 0; i < instance_->numQuadrupeds_; i++) {
        Quadruped* quad = &instance_->quads_[i];
        if (quad->body) {
            const dReal* pos = dBodyGetPosition(quad->body);
            if (std::fabs(pos[0]) <= 60.0 && std::fabs(pos[1]) <= 35.0) {
                countInside++;
            }
        }
    }
    std::snprintf(textBuffer, sizeof(textBuffer), "Quadrupeds in Course: %d", countInside);
    
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    rendering::drawText(textBuffer, 10.0f, 10.0f);
    glPopAttrib();
}
