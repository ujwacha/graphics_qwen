// earth.cpp
// Compile: g++ earth.cpp -o earth -lGL -lGLU -lglut -lm
// Usage: ./earth [texture_path]
// Left-drag = manual free rotation (any direction).
// 'a' = toggle autorotate on/off
// 'p' = toggle autorotate axis perpendicular-to-current (relative to view)
// 'r' = reset
// ESC = quit

#include <GL/glu.h>
#include <GL/glut.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// --------------------- small quaternion helper ---------------------
struct Vec3 {
  float x, y, z;
};
struct Quat {
  float w, x, y, z;
};

static inline Vec3 v3(float x, float y, float z) { return {x, y, z}; }
static inline Vec3 v3norm(Vec3 a) {
  float n = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  if (n <= 1e-8f)
    return v3(1, 0, 0);
  return v3(a.x / n, a.y / n, a.z / n);
}
static inline Vec3 v3cross(Vec3 a, Vec3 b) {
  return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x);
}
static inline float v3dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Quat quat_from_axis_angle(Vec3 axis, float deg) {
  float rad = deg * (3.14159265358979323846f / 180.0f);
  float s = sinf(rad * 0.5f);
  float c = cosf(rad * 0.5f);
  axis = v3norm(axis);
  return {c, axis.x * s, axis.y * s, axis.z * s};
}
static inline Quat quat_mul(Quat a, Quat b) {
  // a * b
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
static inline Quat quat_normalize(Quat q) {
  float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n <= 1e-8f)
    return {1, 0, 0, 0};
  return {q.w / n, q.x / n, q.y / n, q.z / n};
}
static inline Vec3 quat_rotate(Quat q, Vec3 v) {
  // convert v -> quat and compute q * v * q^-1
  Quat qv = {0, v.x, v.y, v.z};
  // q * qv
  Quat t = quat_mul(q, qv);
  // q^-1 (for unit quaternion it's conjugate)
  Quat qc = {q.w, -q.x, -q.y, -q.z};
  Quat r = quat_mul(t, qc);
  return v3(r.x, r.y, r.z);
}
static inline void quat_to_matrix_columnmajor(const Quat &q, float m[16]) {
  // Convert unit quaternion to 4x4 column-major matrix for glMultMatrixf
  float w = q.w, x = q.x, y = q.y, z = q.z;
  float xx = x * x, yy = y * y, zz = z * z;
  float xy = x * y, xz = x * z, yz = y * z;
  float wx = w * x, wy = w * y, wz = w * z;
  // column-major
  m[0] = 1 - 2 * (yy + zz);
  m[4] = 2 * (xy - wz);
  m[8] = 2 * (xz + wy);
  m[12] = 0;
  m[1] = 2 * (xy + wz);
  m[5] = 1 - 2 * (xx + zz);
  m[9] = 2 * (yz - wx);
  m[13] = 0;
  m[2] = 2 * (xz - wy);
  m[6] = 2 * (yz + wx);
  m[10] = 1 - 2 * (xx + yy);
  m[14] = 0;
  m[3] = 0;
  m[7] = 0;
  m[11] = 0;
  m[15] = 1;
}
// -------------------------------------------------------------------

static GLuint earthTex = 0;
static GLUquadric *sphere = nullptr;

// orientation (quaternion)
static Quat orient = {1, 0, 0, 0};

// autorotate settings
static Vec3 autoAxis = {1, 0, 0}; // current autorotation axis (world space)
static float autoSpeedDeg = 0.1f; // degrees per idle tick
static bool autorotateEnabled = true;
static bool perpMode =
    false; // when true, use axis perpendicular to autoAxis relative to viewDir

// interaction
static int lastX = 0, lastY = 0;
static bool dragging = false;
static float zoom = -3.5f;
static int winW = 900, winH = 900;

// simple camera/view direction (we use camera looking down -Z)
static const Vec3 viewDir = {0, 0, -1}; // used to compute perpendicular axis

// texture loader
GLuint loadTexture(const char *filename) {
  stbi_set_flip_vertically_on_load(1);
  int w, h, ch;
  unsigned char *data = stbi_load(filename, &w, &h, &ch, 0);
  if (!data) {
    fprintf(stderr, "Failed to load texture '%s'\n", filename);
    fprintf(stderr, "stbi: %s\n", stbi_failure_reason());
    return 0;
  }
  GLenum fmt = (ch == 4) ? GL_RGBA : GL_RGB;
  GLuint t;
  glGenTextures(1, &t);
  glBindTexture(GL_TEXTURE_2D, t);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  gluBuild2DMipmaps(GL_TEXTURE_2D, fmt, w, h, fmt, GL_UNSIGNED_BYTE, data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  stbi_image_free(data);
  printf("Loaded '%s' (%dx%d) channels=%d\n", filename, w, h, ch);
  return t;
}

void initScene(const char *texfile) {
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_NORMALIZE);
  glShadeModel(GL_SMOOTH);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  GLfloat ambient[] = {0.12f, 0.12f, 0.12f, 1.0f};
  GLfloat diffuse[] = {0.95f, 0.95f, 0.95f, 1.0f};
  GLfloat spec[] = {0.25f, 0.25f, 0.25f, 1.0f};
  GLfloat pos[] = {5.0f, 4.0f, 5.0f, 0.0f};
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, spec);
  glLightfv(GL_LIGHT0, GL_POSITION, pos);

  GLfloat mat[] = {1, 1, 1, 1};
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat);
  GLfloat ms[] = {0.25f, 0.25f, 0.25f, 1.0f};
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, ms);
  GLfloat shin[] = {18.0f};
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin);

  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

  earthTex = loadTexture(texfile);
  if (!earthTex)
    fprintf(stderr, "Warning: texture not loaded.\n");

  sphere = gluNewQuadric();
  gluQuadricNormals(sphere, GLU_SMOOTH);
  gluQuadricTexture(sphere, GL_TRUE);

  Quat initialYaw = quat_from_axis_angle(v3(0, 1, 0), -90.0f);
  orient = quat_normalize(initialYaw);
  autoAxis = v3(0, 0, 1);
}

void reshape(int w, int h) {
  winW = w;
  winH = h;
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(40.0, (double)w / (double)h, 0.1, 100.0);
  glMatrixMode(GL_MODELVIEW);
}

void drawAtmosphere() {
  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_LIGHTING);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDepthMask(GL_FALSE);
  glColor4f(0.08f, 0.12f, 0.22f, 0.16f);
  gluSphere(sphere, 1.06, 64, 64);
  glDepthMask(GL_TRUE);
  glEnable(GL_LIGHTING);
  glPopAttrib();
}

void display() {
  glClearColor(0.01f, 0.02f, 0.04f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glLoadIdentity();
  glTranslatef(0, 0, zoom);

  // apply orientation quaternion as matrix
  float m[16];
  quat_to_matrix_columnmajor(orient, m);
  glMultMatrixf(m);

  // bind texture & draw sphere
  if (earthTex) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, earthTex);
  } else
    glDisable(GL_TEXTURE_2D);

  glColor3f(1, 1, 1);
  gluSphere(sphere, 1.0, 64, 64);
  drawAtmosphere();

  glutSwapBuffers();
}

// compute axis used for autorotation this tick: either autoAxis or
// perpendicular-to-it (relative to viewDir)
Vec3 compute_current_autorotate_axis() {
  if (!perpMode)
    return v3norm(autoAxis);
  // perpendicular axis = normalize(cross(autoAxis, viewDir))
  Vec3 perp = v3cross(autoAxis, viewDir);
  // if too small (parallel), choose an arbitrary perpendicular: cross with
  // world up
  float len = sqrtf(perp.x * perp.x + perp.y * perp.y + perp.z * perp.z);
  if (len < 1e-6f) {
    perp = v3cross(autoAxis, v3(1, 0, 0));
  }
  return v3norm(perp);
}

void idle() {
  if (autorotateEnabled) {
    Vec3 axis = compute_current_autorotate_axis();
    Quat q = quat_from_axis_angle(axis, autoSpeedDeg);
    orient = quat_normalize(quat_mul(q, orient)); // newOrient = q * orient
    // keep autoAxis consistent in world coordinates by rotating it along with
    // orientation so "current" autoAxis = orient * (1,0,0) rotate base axis
    // (1,0,0) by orient to get new autoAxis for next tick
    autoAxis = quat_rotate(orient, v3(1, 0, 0));
    glutPostRedisplay();
  }
}

void mouseButton(int button, int state, int x, int y) {
  if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_DOWN) {
      dragging = true;
      lastX = x;
      lastY = y;
    } else
      dragging = false;
  } else if (button == 3 || button == 4) {
    const float step = 0.3f;
    if (button == 3)
      zoom += step;
    else
      zoom -= step;
    if (zoom > -1.0f)
      zoom = -1.0f;
    if (zoom < -30.0f)
      zoom = -30.0f;
    glutPostRedisplay();
  } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
    orient = {1, 0, 0, 0};
    autoAxis = v3(1, 0, 0);
    zoom = -3.5f;
    glutPostRedisplay();
  }
}

void mouseMove(int x, int y) {
  if (!dragging)
    return;
  int dx = x - lastX;
  int dy = y - lastY;
  lastX = x;
  lastY = y;

  // convert mouse deltas into rotation quaternions
  const float sens = 0.25f; // degrees per pixel
  // yaw around world up (0,1,0)
  Quat qyaw = quat_from_axis_angle(v3(0, 1, 0), dx * sens);
  // pitch around camera-right: camera-right = orient * (1,0,0)
  Vec3 camRight = quat_rotate(orient, v3(1, 0, 0));
  Quat qpitch = quat_from_axis_angle(camRight, dy * sens);

  // apply manual rotation: q_manual * orient
  orient = quat_normalize(quat_mul(qpitch, orient));
  orient = quat_normalize(quat_mul(qyaw, orient));

  // Update autoAxis to remain consistent relative to current orientation:
  autoAxis = quat_rotate(orient, v3(1, 0, 0));

  glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
  if (key == 27)
    exit(0);
  if (key == 'a') {
    autorotateEnabled = !autorotateEnabled;
    printf("Autorotate: %s\n", autorotateEnabled ? "ON" : "OFF");
  }
  if (key == 'p') {
    perpMode = !perpMode;
    printf("Perp mode: %s\n", perpMode ? "ON" : "OFF");
  }
  if (key == 'r') {
    orient = {1, 0, 0, 0};
    autoAxis = v3(1, 0, 0);
    zoom = -3.5f;
    glutPostRedisplay();
  }
  if (key == '+') {
    zoom += 0.3f;
    glutPostRedisplay();
  }
  if (key == '-') {
    zoom -= 0.3f;
    glutPostRedisplay();
  }
}

int main(int argc, char **argv) {
  const char *tex = "earth.png";
  if (argc >= 2)
    tex = argv[1];

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(winW, winH);
  glutCreateWindow(
      "Earth - free rotate + autorotate (toggle perpendicular axis with 'p')");

  initScene(tex);

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouseButton);
  glutMotionFunc(mouseMove);
  glutKeyboardFunc(keyboard);
  glutIdleFunc(idle);

  printf("Controls:\n - Left-drag: free rotation (any direction)\n - Mouse "
         "wheel: zoom\n - 'a': toggle autorotate\n - 'p': toggle "
         "autorotate-perpendicular-mode (axis perpendicular to current auto "
         "axis)\n - 'r': reset\n - ESC: quit\n");

  glutMainLoop();
  return 0;
}
