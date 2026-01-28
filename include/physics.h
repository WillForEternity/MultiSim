#pragma once

// === Physics Utilities ===
// ODE (Open Dynamics Engine) wrapper functions for body/joint creation.

#include <ode/ode.h>
#include "quadruped.h"
#include "config.h"

namespace physics {

/// Initialize ODE world with gravity, contact parameters, and ground plane.
/// @param world Output: created world ID
/// @param space Output: created collision space
/// @param contactGroup Output: created contact joint group
/// @param groundPlane Output: created ground plane geometry
void initWorld(dWorldID& world, dSpaceID& space, dJointGroupID& contactGroup, dGeomID& groundPlane);

/// Clean up all ODE resources.
void cleanup(dWorldID world, dSpaceID space, dJointGroupID contactGroup);

/// Create boundary walls around the simulation area.
/// @param space Collision space to add walls to
/// @param boundaryBoxes Output array of 4 wall geometries
void createBoundaryWalls(dSpaceID space, dGeomID boundaryBoxes[4]);

/// Create the target ball that quadrupeds navigate toward.
/// @param world World to create body in
/// @param space Collision space
/// @param targetBall Output: body ID
/// @param targetBallGeom Output: geometry ID
void createTargetBall(dWorldID world, dSpaceID space, dBodyID& targetBall, dGeomID& targetBallGeom);

/// Create a box rigid body with geometry.
/// @param world World to create body in
/// @param space Collision space (can be nullptr for wheel sub-geoms)
/// @param length Box length (x dimension)
/// @param width Box width (y dimension)  
/// @param height Box height (z dimension)
/// @param mass Total mass of the box
/// @param geomOut Output: created geometry
/// @return Created body ID
dBodyID createBox(dWorldID world, dSpaceID space, 
                  double length, double width, double height, 
                  double mass, dGeomID* geomOut);

/// Create a cylinder rigid body with geometry.
/// @param world World to create body in
/// @param space Collision space
/// @param radius Cylinder radius
/// @param length Cylinder length
/// @param mass Total mass
/// @param isWheel If true, creates a transform geom with rotated cylinder for wheel use
/// @param wheelTransformOut Output: transform geometry (only if isWheel=true)
/// @return Created body ID
dBodyID createCylinder(dWorldID world, dSpaceID space,
                       double radius, double length, double mass,
                       bool isWheel, dGeomID* wheelTransformOut);

/// Create a complete quadruped with body, legs, wheels, and sensors.
/// @param world World to create bodies in
/// @param space Collision space
/// @param quad Quadruped struct to populate
/// @param x Initial x position
/// @param y Initial y position
/// @param simTime Current simulation time
/// @param targetBall Target ball body for distance calculations
/// @param obstacles Array of obstacle geometries (for height checking)
/// @param obstacleSizes Array of obstacle dimensions
/// @param numObstacles Number of obstacles
void createQuadruped(dWorldID world, dSpaceID space,
                     Quadruped* quad, double x, double y,
                     double simTime, dBodyID targetBall,
                     dGeomID* obstacles, dVector3* obstacleSizes, int numObstacles);

/// Destroy all components of a quadruped.
/// @param quad Quadruped to destroy
void destroyQuadruped(Quadruped* quad);

/// Check if a rotation matrix is invalid (NaN or extreme values).
/// @param R 3x3 rotation matrix from ODE
/// @return true if rotation is invalid
bool isBadRotation(const dMatrix3 R);

/// Reset an invalid rotation to identity.
/// @param body Body to fix rotation on
void safeNormalizeBodyRotation(dBodyID body);

} // namespace physics
