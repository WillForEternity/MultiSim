// rendering.cpp
// OpenGL and DrawStuff visualization functions.

#define GL_SILENCE_DEPRECATION

#include "rendering.h"
#include "config.h"
#include <drawstuff/drawstuff.h>
#include <GLUT/glut.h>
#include <cmath>

namespace rendering {

// Helper functions to convert dReal (double) to float arrays for DrawStuff
static void toFloat3(const dReal* src, float dst[3]) {
    dst[0] = static_cast<float>(src[0]);
    dst[1] = static_cast<float>(src[1]);
    dst[2] = static_cast<float>(src[2]);
}

static void toFloat12(const dReal* src, float dst[12]) {
    for (int i = 0; i < 12; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
}

void hsv2rgb(double h, double s, double v, float& r, float& g, float& b) {
    double c = v * s;
    double h_prime = std::fmod(h / 60.0, 6);
    double x = c * (1 - std::fabs(std::fmod(h_prime, 2) - 1));
    double m = v - c;
    
    double r1, g1, b1;
    if (0 <= h_prime && h_prime < 1) {
        r1 = c; g1 = x; b1 = 0;
    } else if (1 <= h_prime && h_prime < 2) {
        r1 = x; g1 = c; b1 = 0;
    } else if (2 <= h_prime && h_prime < 3) {
        r1 = 0; g1 = c; b1 = x;
    } else if (3 <= h_prime && h_prime < 4) {
        r1 = 0; g1 = x; b1 = c;
    } else if (4 <= h_prime && h_prime < 5) {
        r1 = x; g1 = 0; b1 = c;
    } else {
        r1 = c; g1 = 0; b1 = x;
    }
    
    r = static_cast<float>(r1 + m);
    g = static_cast<float>(g1 + m);
    b = static_cast<float>(b1 + m);
}

void drawText(const char* text, float x, float y) {
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

void drawObstacles(dGeomID* obstacles, dVector3* sizes, int count) {
    dsSetTexture(DS_SKY);
    dsSetColor(1.0, 1.0, 1.0);
    
    for (int i = 0; i < count; i++) {
        float fpos[3], frot[12], fsizes[3];
        toFloat3(dGeomGetPosition(obstacles[i]), fpos);
        toFloat12(dGeomGetRotation(obstacles[i]), frot);
        toFloat3(sizes[i], fsizes);
        dsDrawBox(fpos, frot, fsizes);
    }
}

void drawObstacleLines(dGeomID* obstacles, dVector3* sizes, int count) {
    glDisable(GL_LIGHTING);
    glLineWidth(8.0);
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    
    for (int i = 0; i < count; i++) {
        const dReal* pos = dGeomGetPosition(obstacles[i]);
        double sy = sizes[i][1] / 2.0;
        double x = pos[0] + 0.5;
        double y1 = pos[1] - sy - 2.0;
        double y2 = pos[1] + sy + 2.0;
        glVertex3f(static_cast<float>(x), static_cast<float>(y1), 0.0f);
        glVertex3f(static_cast<float>(x), static_cast<float>(y2), 0.0f);
    }
    
    glEnd();
    glEnable(GL_LIGHTING);
}

void drawBoundaryWalls(dGeomID boundaryBoxes[4]) {
    using namespace config;
    
    dsSetColor(1.0, 1.0, 1.0);
    
    for (int i = 0; i < 4; i++) {
        const dReal* pos = dGeomGetPosition(boundaryBoxes[i]);
        const dReal* rot = dGeomGetRotation(boundaryBoxes[i]);
        
        dVector3 dims;
        if (i < 2) {
            // Horizontal walls (top/bottom)
            dims[0] = (WORLD_X_MAX - WORLD_X_MIN) + 1.0;
            dims[1] = 1.0;
            dims[2] = 5.0;
        } else {
            // Vertical walls (left/right)
            dims[0] = 1.0;
            dims[1] = (WORLD_Y_MAX - WORLD_Y_MIN) + 1.0;
            dims[2] = 5.0;
        }
        float fpos[3], frot[12], fdims[3];
        toFloat3(pos, fpos);
        toFloat12(rot, frot);
        toFloat3(dims, fdims);
        dsDrawBox(fpos, frot, fdims);
    }
}

void drawQuadruped(const Quadruped* quad, double simTime) {
    using namespace config;
    
    if (!quad->body) return;
    
    const dReal* bodyPos = dBodyGetPosition(quad->body);
    
    // Determine color based on blink state
    float color[3] = {1.0f, 1.0f, 1.0f};
    double elapsed = simTime - quad->blinkStartTime;
    
    if (quad->blinkCount > 0 && elapsed < quad->blinkCount * BLINK_PERIOD) {
        if (std::fmod(elapsed, BLINK_PERIOD) < (BLINK_PERIOD / 2)) {
            switch (quad->blinkType) {
                case BLINK_GREEN:
                    color[0] = 0.0f; color[1] = 1.0f; color[2] = 0.0f;
                    break;
                case BLINK_RED:
                    color[0] = 1.0f; color[1] = 0.0f; color[2] = 0.0f;
                    break;
                case BLINK_BLUE:
                    color[0] = 0.0f; color[1] = 0.0f; color[2] = 1.0f;
                    break;
            }
        }
    }
    
    dsSetColor(color[0], color[1], color[2]);
    dsSetTexture(DS_WOOD);
    
    // Draw body
    dVector3 bodySides = {BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT};
    float fbodyPos[3], fbodyRot[12], fbodySides[3];
    toFloat3(bodyPos, fbodyPos);
    toFloat12(dBodyGetRotation(quad->body), fbodyRot);
    toFloat3(bodySides, fbodySides);
    dsDrawBox(fbodyPos, fbodyRot, fbodySides);
    
    // Draw legs and wheels
    for (int j = 0; j < 4; j++) {
        if (quad->leg[j]) {
            dVector3 legSides = {LEG_WIDTH, LEG_WIDTH, LEG_LENGTH};
            float flegPos[3], flegRot[12], flegSides[3];
            toFloat3(dBodyGetPosition(quad->leg[j]), flegPos);
            toFloat12(dBodyGetRotation(quad->leg[j]), flegRot);
            toFloat3(legSides, flegSides);
            dsDrawBox(flegPos, flegRot, flegSides);
        }
        if (quad->wheel[j]) {
            float fwheelPos[3], fwheelRot[12];
            toFloat3(dBodyGetPosition(quad->wheel[j]), fwheelPos);
            toFloat12(dBodyGetRotation(quad->wheel[j]), fwheelRot);
            dsDrawCylinder(fwheelPos, fwheelRot, 
                          static_cast<float>(WHEEL_RADIUS), static_cast<float>(WHEEL_WIDTH));
        }
    }
}

void drawSensorRays(const Quadruped* quad) {
    using namespace config;
    
    if (!quad->body) return;
    
    const dReal* bodyPos = dBodyGetPosition(quad->body);
    double baseAngle = quad->orientation;
    
    for (int i = 0; i < NUM_RAYS; i++) {
        double angle = baseAngle + (2 * M_PI * i / NUM_RAYS);
        dVector3 sensorOrigin = {bodyPos[0], bodyPos[1], bodyPos[2]};
        dVector3 sensorDir = {std::cos(angle), std::sin(angle), 0.0};
        
        // Color gradient: red (0) = close, blue (240) = far
        double hue = quad->sensorValues[i] * 240.0;
        float r, g, b;
        hsv2rgb(hue, 1.0, 1.0, r, g, b);
        
        double drawLength = DRAWN_MAX_LENGTH * quad->sensorValues[i];
        glColor3f(r, g, b);
        
        glVertex3d(sensorOrigin[0], sensorOrigin[1], sensorOrigin[2]);
        glVertex3d(sensorOrigin[0] + sensorDir[0] * drawLength,
                   sensorOrigin[1] + sensorDir[1] * drawLength,
                   sensorOrigin[2] + sensorDir[2] * drawLength);
    }
}

void drawAllSensorRays(Quadruped* quads, int count) {
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glLineWidth(1.25);
    glBegin(GL_LINES);
    
    for (int q = 0; q < count; q++) {
        drawSensorRays(&quads[q]);
    }
    
    glEnd();
    glPopAttrib();
}

void drawTargetBall(dBodyID targetBall) {
    dsSetTexture(DS_SKY);
    dsSetColor(0.0, 1.0, 0.0);
    
    float ftPos[3], ftRot[12];
    toFloat3(dBodyGetPosition(targetBall), ftPos);
    toFloat12(dBodyGetRotation(targetBall), ftRot);
    dsDrawSphere(ftPos, ftRot, static_cast<float>(config::TARGET_RADIUS));
}

void drawGround() {
    #if USE_TEXTURE
    dsSetTexture(DS_CHECKERED);
    #endif
    dsSetColor(1.0, 1.0, 1.0);
    
    float groundSize[3] = {10.0f, 10.0f, 0.01f};
    float groundPos[3] = {0.0f, 0.0f, 0.0f};
    float groundRot[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    dsDrawBox(groundPos, groundRot, groundSize);
    
    dsSetTexture(DS_GROUND);
}

} // namespace rendering
