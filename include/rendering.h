#pragma once

// === Rendering Utilities ===
// OpenGL and DrawStuff visualization functions.

#include <ode/ode.h>
#include "quadruped.h"
#include "config.h"

namespace rendering {

/// Convert HSV color to RGB.
/// @param h Hue in [0, 360]
/// @param s Saturation in [0, 1]
/// @param v Value in [0, 1]
/// @param r Output red in [0, 1]
/// @param g Output green in [0, 1]
/// @param b Output blue in [0, 1]
void hsv2rgb(double h, double s, double v, float& r, float& g, float& b);

/// Draw 2D text overlay on screen.
/// @param text Text string to draw
/// @param x Screen x position
/// @param y Screen y position
void drawText(const char* text, float x, float y);

/// Draw all obstacles in the simulation.
/// @param obstacles Array of obstacle geometries
/// @param sizes Array of obstacle dimensions (length, width, height)
/// @param count Number of obstacles
void drawObstacles(dGeomID* obstacles, dVector3* sizes, int count);

/// Draw obstacle guidance lines on the ground.
/// @param obstacles Array of obstacle geometries
/// @param sizes Array of obstacle dimensions
/// @param count Number of obstacles
void drawObstacleLines(dGeomID* obstacles, dVector3* sizes, int count);

/// Draw boundary walls around the simulation.
/// @param boundaryBoxes Array of 4 wall geometries
void drawBoundaryWalls(dGeomID boundaryBoxes[4]);

/// Draw a single quadruped with visual feedback.
/// @param quad Quadruped to draw
/// @param simTime Current simulation time (for blink animations)
void drawQuadruped(const Quadruped* quad, double simTime);

/// Draw sensor rays for a quadruped with color-coded distances.
/// @param quad Quadruped whose sensors to draw
void drawSensorRays(const Quadruped* quad);

/// Draw all sensor rays for multiple quadrupeds.
/// @param quads Array of quadrupeds
/// @param count Number of quadrupeds
void drawAllSensorRays(Quadruped* quads, int count);

/// Draw the target ball.
/// @param targetBall Body ID of target
void drawTargetBall(dBodyID targetBall);

/// Draw the ground plane.
void drawGround();

} // namespace rendering
