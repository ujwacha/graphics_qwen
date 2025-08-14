// earth_with_menu.cpp
// Compile:
// g++ earth_with_menu.cpp -o earth_with_menu -lGL -lGLU -lglut -lm -std=c++11
// Usage:
// ./earth_with_menu [color_texture] [heightmap]
// Defaults: color = "earth.png", height = "earth_elevation_grayscale.png"
// Right-click inside the window to open a menu and change the color texture.

#include <GL/glu.h>
#include <GL/glut.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// --- small vector/quaternion helpers (same as before) ---
struct Vec3 { float x,y,z; };
struct Quat { float w,x,y,z; };
static inline Vec3 v3(float x,float y,float z){ return {x,y,z}; }
static inline Vec3 v3norm(Vec3 a){
  float n = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
  if(n<=1e-8f) return v3(1,0,0);
  return v3(a.x/n, a.y/n, a.z/n);
}
static inline Vec3 v3cross(Vec3 a, Vec3 b){
  return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline Quat quat_from_axis_angle(Vec3 axis, float deg){
  float rad = deg * (3.14159265358979323846f/180.0f);
  float s = sinf(rad*0.5f), c = cosf(rad*0.5f);
  axis = v3norm(axis);
  return {c, axis.x*s, axis.y*s, axis.z*s};
}
static inline Quat quat_mul(Quat a, Quat b){
  return {
    a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
    a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
    a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
  };
}
static inline Quat quat_normalize(Quat q){
  float n = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
  if(n<=1e-8f) return {1,0,0,0};
  return {q.w/n, q.x/n, q.y/n, q.z/n};
}
static inline Vec3 quat_rotate(Quat q, Vec3 v){
  Quat qv = {0,v.x,v.y,v.z};
  Quat t = quat_mul(q,qv);
  Quat qc = {q.w,-q.x,-q.y,-q.z};
  Quat r = quat_mul(t,qc);
  return v3(r.x,r.y,r.z);
}
static inline void quat_to_matrix_columnmajor(const Quat &q, float m[16]){
  float w=q.w,x=q.x,y=q.y,z=q.z;
  float xx=x*x, yy=y*y, zz=z*z;
  float xy=x*y, xz=x*z, yz=y*z;
  float wx=w*x, wy=w*y, wz=w*z;
  m[0]=1-2*(yy+zz); m[4]=2*(xy-wz); m[8]=2*(xz+wy); m[12]=0;
  m[1]=2*(xy+wz); m[5]=1-2*(xx+zz); m[9]=2*(yz-wx); m[13]=0;
  m[2]=2*(xz-wy); m[6]=2*(yz+wx); m[10]=1-2*(xx+yy); m[14]=0;
  m[3]=0; m[7]=0; m[11]=0; m[15]=1;
}
// -------------------------------------------------------------------

static GLuint earthTex = 0;
static GLUquadric *quadSphere = nullptr;

// orientation (quaternion)
static Quat orient = {1, 0, 0, 0};

// autorotate settings
static Vec3 autoAxis = {1, 0, 0};
static float autoSpeedDeg = 0.1f;
static bool autorotateEnabled = true;
static bool perpMode = false;

// interaction
static int lastX = 0, lastY = 0;
static bool dragging = false;
static float zoom = -3.5f;
static int winW = 900, winH = 900;

// simple camera/view direction (we use camera looking down -Z)
static const Vec3 viewDir = {0, 0, -1};

// heightmap resources (unchanged)
static unsigned char *heightImg = nullptr;
static int hW = 0, hH = 0, hCh = 0;
static bool heightmapEnabled = true;
static float heightScale = 0.02f; // displacement magnitude

// orientation toggles for height sampling (keeps prior fixes)
static bool flipHeightV = false;
static bool flipHeightU = false;
static bool transposeHeight = false;
static float uOffset = 0.0f;
static bool smoothingEnabled = true;

// mesh storage
static std::vector<float> vdata; // x,y,z
static std::vector<float> ndata; // nx,ny,nz
static std::vector<float> tdata; // u,v
static std::vector<unsigned int> idata; // triangles

static int meshStacks = 1024;
static int meshSlices = 1024;

// ----------------- NEW: list of candidate color textures -----------------
// Edit/add filenames here (these are the PNGs you showed in your directory).
// Exclude the heightmap file.
static std::vector<std::string> colorTextureFiles = {
  "earth.png",
  "earth.png.bak",
  "DayTemp.png",
  "Rainfall.png",
  "SeaSurfaceTemp.png",
  "LeafAreaIndex.png"
};
// -------------------------------------------------------------------------

// texture loader (do not flip — we control texcoords)
GLuint loadTexture(const char *filename) {
  stbi_set_flip_vertically_on_load(0);
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

// heightmap loader (unchanged)
unsigned char *loadHeightImage(const char *filename, int &w, int &h, int &ch) {
  stbi_set_flip_vertically_on_load(0);
  unsigned char *data = stbi_load(filename, &w, &h, &ch, 0);
  if (!data) {
    fprintf(stderr, "Failed to load heightmap '%s'\n", filename);
    fprintf(stderr, "stbi: %s\n", stbi_failure_reason());
  } else {
    printf("Loaded heightmap '%s' (%dx%d) channels=%d\n", filename, w, h, ch);
  }
  return data;
}

// sample helpers (same as last working version)
float sampleHeightUV_raw(float u, float v);
float sampleHeightUV(float u, float v);
Vec3 posOnSphereUV(float u, float v, float r, float scale);
void generateSphereMesh(int stacks, int slices, float radius, float scale);

// init (only difference: initial color texture can be chosen by argv)
void initScene(const char *texfile, const char *heightfile) {
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

  // load initial color texture
  earthTex = loadTexture(texfile);
  if (!earthTex)
    fprintf(stderr, "Warning: color texture not loaded.\n");

  // load heightmap
  if (heightImg) {
    stbi_image_free(heightImg);
    heightImg = nullptr;
  }
  heightImg = loadHeightImage(heightfile, hW, hH, hCh);
  if (!heightImg) {
    fprintf(stderr, "Warning: heightmap not loaded; heightmap disabled.\n");
    heightmapEnabled = false;
  }

  quadSphere = gluNewQuadric();
  gluQuadricNormals(quadSphere, GLU_SMOOTH);
  gluQuadricTexture(quadSphere, GL_TRUE);

  Quat initialYaw = quat_from_axis_angle(v3(0, 1, 0), -90.0f);
  orient = quat_normalize(initialYaw);
  autoAxis = v3(0, 0, 1);

  // generate mesh
  generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
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
  gluSphere(quadSphere, 1.06, 64, 64);
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

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);

  glVertexPointer(3, GL_FLOAT, 0, vdata.data());
  glNormalPointer(GL_FLOAT, 0, ndata.data());
  glTexCoordPointer(2, GL_FLOAT, 0, tdata.data());

  glColor3f(1, 1, 1);
  glDrawElements(GL_TRIANGLES, (GLsizei)idata.size(), GL_UNSIGNED_INT, idata.data());

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);

  drawAtmosphere();

  glutSwapBuffers();
}

// --- the sampleHeightUV_raw/sampleHeightUV/posOnSphereUV/generateSphereMesh functions ---
// (copying the same implementations from the previous corrected version so normals & smoothing work)
float sampleHeightUV_raw(float u, float v) {
  if (!heightImg || hW == 0 || hH == 0) return 0.5f;
  while (u < 0.0f) u += 1.0f;
  while (u >= 1.0f) u -= 1.0f;
  if (v < 0.0f) v = 0.0f;
  if (v > 1.0f) v = 1.0f;
  float fx = u * (hW - 1);
  float fy = v * (hH - 1);
  int x0 = (int)floorf(fx);
  int y0 = (int)floorf(fy);
  int x1 = x0 + 1; if (x1 >= hW) x1 = 0;
  int y1 = y0 + 1; if (y1 >= hH) y1 = hH - 1;
  float sx = fx - x0, sy = fy - y0;
  auto samp = [&](int x, int y)->float{
    int idx = (y * hW + x) * hCh;
    if (hCh == 1) return heightImg[idx] / 255.0f;
    float r = heightImg[idx+0] / 255.0f;
    float g = heightImg[idx+1] / 255.0f;
    float b = heightImg[idx+2] / 255.0f;
    return 0.2126f*r + 0.7152f*g + 0.0722f*b;
  };
  float h00 = samp(x0,y0), h10 = samp(x1,y0), h01 = samp(x0,y1), h11 = samp(x1,y1);
  float h0 = h00*(1 - sx) + h10*sx;
  float h1 = h01*(1 - sx) + h11*sx;
  return h0*(1 - sy) + h1*sy;
}

float sampleHeightUV(float u_in, float v_in) {
  if (!heightImg || hW == 0 || hH == 0) return 0.5f;
  float u = u_in, v = v_in;
  if (transposeHeight) { float tmp = u; u = v; v = tmp; }
  if (flipHeightU) u = 1.0f - u;
  if (flipHeightV) v = 1.0f - v;
  u += uOffset;
  while (u < 0.0f) u += 1.0f;
  while (u >= 1.0f) u -= 1.0f;
  if (!smoothingEnabled) return sampleHeightUV_raw(u, v);
  float du = 1.0f / (float)meshSlices;
  float dv = 1.0f / (float)meshStacks;
  float hC = sampleHeightUV_raw(u, v);
  float hL = sampleHeightUV_raw(u - du, v);
  float hR = sampleHeightUV_raw(u + du, v);
  float hU = sampleHeightUV_raw(u, v - dv);
  float hD = sampleHeightUV_raw(u, v + dv);
  return (hC + hL + hR + hU + hD) / 5.0f;
}

Vec3 posOnSphereUV(float u, float v, float r, float scale) {
  float theta = u * 2.0f * 3.14159265358979323846f;
  float phi = v * 3.14159265358979323846f;
  float sinphi = sinf(phi), cosphi = cosf(phi);
  float x = sinphi * cosf(theta);
  float y = cosphi;
  float z = sinphi * sinf(theta);
  float h = sampleHeightUV(u, v);
  float disp = (h - 0.5f) * 2.0f * scale;
  float rad = r + disp;
  return v3(x * rad, y * rad, z * rad);
}

void generateSphereMesh(int stacks, int slices, float radius, float scale) {
  vdata.clear(); ndata.clear(); tdata.clear(); idata.clear();
  vdata.reserve((stacks + 1) * (slices + 1) * 3);
  ndata.reserve((stacks + 1) * (slices + 1) * 3);
  tdata.reserve((stacks + 1) * (slices + 1) * 2);
  for (int i = 0; i <= stacks; ++i) {
    float v = (float)i / (float)stacks;
    for (int j = 0; j <= slices; ++j) {
      float u = (float)j / (float)slices;
      Vec3 p = posOnSphereUV(u, v, radius, heightmapEnabled ? scale : 0.0f);
      vdata.push_back(p.x); vdata.push_back(p.y); vdata.push_back(p.z);
      // keep color texture top -> north pole mapping
      tdata.push_back(u); tdata.push_back(1.0f - v);
      ndata.push_back(0.0f); ndata.push_back(0.0f); ndata.push_back(0.0f);
    }
  }
  for (int i = 0; i < stacks; ++i) {
    for (int j = 0; j < slices; ++j) {
      unsigned int row1 = i * (slices + 1);
      unsigned int row2 = (i + 1) * (slices + 1);
      unsigned int a = row1 + j;
      unsigned int b = row2 + j;
      unsigned int c = row2 + (j + 1);
      unsigned int d = row1 + (j + 1);
      idata.push_back(a); idata.push_back(b); idata.push_back(c);
      idata.push_back(a); idata.push_back(c); idata.push_back(d);
    }
  }
  // normals via central differences
  float du = 1.0f / (float)slices;
  float dv = 1.0f / (float)stacks;
  int idx = 0;
  for (int i = 0; i <= stacks; ++i) {
    float v = (float)i / (float)stacks;
    for (int j = 0; j <= slices; ++j) {
      float u = (float)j / (float)slices;
      float u_p = u + du; if (u_p >= 1.0f) u_p -= 1.0f;
      float u_m = u - du; if (u_m < 0.0f) u_m += 1.0f;
      float v_p = v + dv; if (v_p > 1.0f) v_p = 1.0f;
      float v_m = v - dv; if (v_m < 0.0f) v_m = 0.0f;
      Vec3 p  = posOnSphereUV(u,   v,   radius, heightmapEnabled ? scale : 0.0f);
      Vec3 pu = posOnSphereUV(u_p, v,   radius, heightmapEnabled ? scale : 0.0f);
      Vec3 pm = posOnSphereUV(u_m, v,   radius, heightmapEnabled ? scale : 0.0f);
      Vec3 pv = posOnSphereUV(u,   v_p, radius, heightmapEnabled ? scale : 0.0f);
      Vec3 pw = posOnSphereUV(u,   v_m, radius, heightmapEnabled ? scale : 0.0f);
      Vec3 t1 = v3(pu.x - pm.x, pu.y - pm.y, pu.z - pm.z);
      Vec3 t2 = v3(pv.x - pw.x, pv.y - pw.y, pv.z - pw.z);
      Vec3 n = v3cross(t1, t2);
      n = v3norm(n);
      float nl = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
      if (nl < 1e-6f) n = v3norm(p);
      ndata[idx*3 + 0] = n.x;
      ndata[idx*3 + 1] = n.y;
      ndata[idx*3 + 2] = n.z;
      ++idx;
    }
  }
  printf("Generated mesh: stacks=%d slices=%d verts=%zu tris=%zu\n", stacks, slices, vdata.size()/3, idata.size()/3);
}

// autorotate helpers (unchanged)
Vec3 compute_current_autorotate_axis() {
  if (!perpMode) return v3norm(autoAxis);
  Vec3 perp = v3cross(autoAxis, viewDir);
  float len = sqrtf(perp.x * perp.x + perp.y * perp.y + perp.z * perp.z);
  if (len < 1e-6f) perp = v3cross(autoAxis, v3(1, 0, 0));
  return v3norm(perp);
}

void idle() {
  if (autorotateEnabled) {
    Vec3 axis = compute_current_autorotate_axis();
    Quat q = quat_from_axis_angle(axis, autoSpeedDeg);
    orient = quat_normalize(quat_mul(q, orient));
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
    if (zoom > -1.0f) zoom = -1.0f;
    if (zoom < -30.0f) zoom = -30.0f;
    glutPostRedisplay();
  } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
    // keep right-click for popup menu (do not reset)
  }
}

void mouseMove(int x, int y) {
  if (!dragging) return;
  int dx = x - lastX, dy = y - lastY;
  lastX = x; lastY = y;
  const float sens = 0.25f;
  Quat qyaw = quat_from_axis_angle(v3(0,1,0), dx * sens);
  Vec3 camRight = quat_rotate(orient, v3(1, 0, 0));
  Quat qpitch = quat_from_axis_angle(camRight, dy * sens);
  orient = quat_normalize(quat_mul(qpitch, orient));
  orient = quat_normalize(quat_mul(qyaw, orient));
  autoAxis = quat_rotate(orient, v3(1, 0, 0));
  glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
  if (key == 27) exit(0);
  if (key == 'a') {
    autorotateEnabled = !autorotateEnabled;
    printf("Autorotate: %s\n", autorotateEnabled ? "ON" : "OFF");
  } else if (key == 'p') {
    perpMode = !perpMode;
    printf("Perp mode: %s\n", perpMode ? "ON" : "OFF");
  } else if (key == 'r') {
    orient = {1, 0, 0, 0};
    autoAxis = v3(1, 0, 0);
    zoom = -3.5f;
    glutPostRedisplay();
  } else if (key == '+') {
    zoom += 0.3f;
    glutPostRedisplay();
  } else if (key == '-') {
    zoom -= 0.3f;
    glutPostRedisplay();
  } else if (key == 'm') {
    heightmapEnabled = !heightmapEnabled;
    printf("Heightmap: %s\n", heightmapEnabled ? "ENABLED" : "DISABLED");
    generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
    glutPostRedisplay();
  } else if (key == '[') {
    heightScale *= 0.8f;
    printf("Height scale = %g\n", heightScale);
    generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
    glutPostRedisplay();
  } else if (key == ']') {
    heightScale *= 1.25f;
    printf("Height scale = %g\n", heightScale);
    generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
    glutPostRedisplay();
  } else if (key == 'v') {
    flipHeightV = !flipHeightV;
    printf("Flip height V: %s\n", flipHeightV ? "ON" : "OFF");
    generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
    glutPostRedisplay();
  } else if (key == 'u') {
    flipHeightU = !flipHeightU;
    printf("Flip height U: %s\n", flipHeightU ? "ON" : "OFF");
    generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
    glutPostRedisplay();
  } else if (key == 't') {
    transposeHeight = !transposeHeight;
    printf("Transpose height: %s\n", transposeHeight ? "ON" : "OFF");
    generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
    glutPostRedisplay();
  } else if (key == 'o') {
    uOffset += 0.5f; if (uOffset >= 1.0f) uOffset -= 1.0f;
    printf("Longitude offset = %g\n", uOffset);
    generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
    glutPostRedisplay();
  } else if (key == 's') {
    smoothingEnabled = !smoothingEnabled;
    printf("Smoothing: %s\n", smoothingEnabled ? "ON" : "OFF");
    generateSphereMesh(meshStacks, meshSlices, 1.0f, heightScale);
    glutPostRedisplay();
  }
}

// ----------------- NEW: menu handling -----------------
void setColorTextureFromFile(const char *filename) {
  // delete previous GL texture
  if (earthTex) {
    glDeleteTextures(1, &earthTex);
    earthTex = 0;
  }
  earthTex = loadTexture(filename);
  if (!earthTex) fprintf(stderr, "Failed to load color texture '%s'\n", filename);
  glutPostRedisplay();
}

void textureMenuHandler(int id) {
  // ID maps to index in colorTextureFiles; -1 = Reset, -2 = Quit
  if (id >= 0 && id < (int)colorTextureFiles.size()) {
    printf("Switching texture -> %s\n", colorTextureFiles[id].c_str());
    setColorTextureFromFile(colorTextureFiles[id].c_str());
  } else if (id == -1) {
    // reset to default earth.png if present
    printf("Resetting texture to 'earth.png'\n");
    setColorTextureFromFile("earth.png");
  } else if (id == -2) {
    exit(0);
  }
}

void createTextureMenu() {
  int menu = glutCreateMenu(textureMenuHandler);
  // add entries for each texture
  for (int i = 0; i < (int)colorTextureFiles.size(); ++i) {
    glutAddMenuEntry(colorTextureFiles[i].c_str(), i);
  }
  glutAddMenuEntry("Reset to earth.png", -1);
  glutAddMenuEntry("Quit", -2);
  glutAttachMenu(GLUT_RIGHT_BUTTON); // right-click to open
}
// -------------------------------------------------------

int main(int argc, char **argv) {
  const char *colorTex = "earth.png";
  const char *heightTex = "earth_elevation_grayscale.png";
  if (argc >= 2) colorTex = argv[1];
  if (argc >= 3) heightTex = argv[2];

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(winW, winH);
  glutCreateWindow("Earth with menu - right-click to pick texture");

  initScene(colorTex, heightTex);

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouseButton);
  glutMotionFunc(mouseMove);
  glutKeyboardFunc(keyboard);
  glutIdleFunc(idle);

  createTextureMenu(); // build and attach menu

  printf("Controls:\n - Left-drag: free rotation (any direction)\n - Mouse wheel: zoom\n - Right-click: texture menu\n - 'a': toggle autorotate\n - 'p': toggle autorotate-perpendicular-mode\n - 'r': reset\n - 'm': toggle heightmap\n - '[' / ']': decrease / increase height exaggeration\n - 'v','u','t','o','s' : heightmap orientation/smoothing toggles\n - ESC: quit\n");

  glutMainLoop();
  return 0;
}
