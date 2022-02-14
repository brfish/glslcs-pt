#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "stb/stb_image_write.h"

static constexpr uint32_t kDefaultImageWidth       = 1024;
static constexpr uint32_t kDefaultImageHeight      = 768;
static constexpr int32_t  kDefaultSpp              = 256;
static constexpr int32_t  kDefaultMaxBounce        = 8;
static constexpr int32_t  kDefaultRRStartBounce    = 4;

static const std::string kDefaultSceneFile   = "scene/cornell.txt";
static const std::string kDefaultOutputFile  = "image.exr";

constexpr uint32_t kImageWidth   = 1024;
constexpr uint32_t kImageHeight  = 768;
constexpr uint32_t kWindowWidth  = 1024;
constexpr uint32_t kWindowHeight = 768;

constexpr uint32_t kTileWidth    = 128;
constexpr uint32_t kTileHeight   = 128;

const std::string kCSPath = "shader/glslcs_pt.comp.glsl";
const std::string kVSPath = "shader/display.vert.glsl";
const std::string kFSPath = "shader/display.frag.glsl";

static const float kQuadVertices[] = {
     1.0F,  1.0F, 0.0F,   1.0F, 1.0F,
     1.0F, -1.0F, 0.0F,   1.0F, 0.0F,
    -1.0F, -1.0F, 0.0F,   0.0F, 0.0F,
    -1.0F,  1.0F, 0.0F,   0.0F, 1.0F 
};

static const uint16_t kQuadIndices[] = {
    0, 1, 3,
    1, 2, 3
};

static inline GLuint createShader(GLenum shaderType, const std::string &filePath) {
    std::ifstream input {filePath};
    std::string src {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    input.close();
    const GLchar *data = src.c_str();
    GLint length = static_cast<GLint>(src.size());
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &data, &length);
    glCompileShader(shader);
    GLint compileStatus;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    if (compileStatus != GL_TRUE) {
        GLchar buf[512];
        GLsizei infoLength;
        glGetShaderInfoLog(shader, 512, &infoLength, buf);
        std::fprintf(stderr, buf);
        std::abort();
    }
    return shader;
}

class Shader {
public:
    Shader(const std::string &vsPath, const std::string &fsPath) {
        GLuint vs = createShader(GL_VERTEX_SHADER, vsPath);
        GLuint fs = createShader(GL_FRAGMENT_SHADER, fsPath);
        id_ = glCreateProgram();
        glAttachShader(id_, vs);
        glAttachShader(id_, fs);
        glLinkProgram(id_);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint status;
        glGetProgramiv(id_, GL_LINK_STATUS, &status);
        if (status != GL_TRUE) {
            GLchar buf[512];
            GLsizei infoLength;
            glGetProgramInfoLog(id_, 512, &infoLength, buf);
            std::fprintf(stderr, buf);
            std::abort();
        }
    }

    GLuint id() const {
        return id_;
    }

    operator GLuint() const {
        return id_;
    }

private:
    GLuint id_;
};

class ComputeShader {
public:
    ComputeShader(const std::string &csPath) {
        GLuint cs = createShader(GL_COMPUTE_SHADER, csPath);
        id_ = glCreateProgram();
        glAttachShader(id_, cs);
        glLinkProgram(id_);
        glDeleteShader(cs);
        GLint status;
        glGetProgramiv(id_, GL_LINK_STATUS, &status);
        if (status != GL_TRUE) {
            GLchar buf[512];
            GLsizei infoLength;
            glGetProgramInfoLog(id_, 512, &infoLength, buf);
            std::fprintf(stderr, buf);
            std::abort();
        }
    }

    GLuint id() const {
        return id_;
    }

    operator GLuint() const {
        return id_;
    }

private:
    GLuint id_;
};

struct Camera {
    glm::vec3   position;
    glm::vec3   gaze;
    float       fov;
    glm::uvec2  resolution;
};

enum class SurfaceType : uint32_t {
    eDiffuse                = 1,
    eFresnelReflection      = 2,
    eFresnelTransmission    = 4,
    eFresnelDielectric      = 6,
    eMirror                 = 8,
};

struct Material {
    glm::vec3   color;
    float       _padding0; // For std430 layout.
    glm::vec3   emission;
    SurfaceType surfaceType;  
};

struct Sphere {
    glm::vec3   position;
    float       radius;
    Material    material;
};

struct Scene {
    std::vector<Sphere>     objects;
    std::vector<int32_t>    lights;
    Camera                  mainCamera;
    uint32_t                targetWidth;
    uint32_t                targetHeight;
};

struct Config {
    uint32_t    outputWidth;
    uint32_t    outputHeight;
    std::string outputFile;
    std::string sceneFile;
    int32_t     maxBounce;
    bool        enableRR;
    int32_t     rrStartBounce;
    int32_t     spp;
};

static void uniformCamera(const ComputeShader &shader, const Camera &camera) {
    GLint loc;
    loc = glGetUniformLocation(shader, "uCamera.position");
    glUniform3fv(loc, 1, glm::value_ptr(camera.position));
    loc = glGetUniformLocation(shader, "uCamera.gaze");
    glUniform3fv(loc, 1, glm::value_ptr(camera.gaze));
    loc = glGetUniformLocation(shader, "uCamera.fov");
    glUniform1f(loc, camera.fov);
    loc = glGetUniformLocation(shader, "uCamera.resolution");
    glUniform2uiv(loc, 1, glm::value_ptr(camera.resolution));
}

static void uniformConfig(const ComputeShader &shader, const Config &config) {
    GLint loc;
    loc = glGetUniformLocation(shader, "uMaxBounce");
    glUniform1i(loc, config.maxBounce);
    loc = glGetUniformLocation(shader, "uRRStartBounce");
    glUniform1i(loc, config.enableRR ? config.rrStartBounce : -1);
    loc = glGetUniformLocation(shader, "uSpp");
    glUniform1i(loc, config.spp);
}

static void loadScene(const std::string &path, Scene &scene) {
    std::ifstream input {path};
    std::string line;
    scene.objects.clear();
    scene.lights.clear();
    scene.lights.push_back(0);
    while (std::getline(input, line)) {
        size_t p = line.find_first_not_of(' ');
        if (line[p] == '#' || line.empty())
            continue;
        std::stringstream ls {line};
        std::string type;
        ls >> type;
        if (type == "target") {
            uint32_t w, h;
            ls >> w >> h;
            scene.targetWidth = w;
            scene.targetHeight = h;
        } else if (type == "camera") {
            float x, y, z;
            ls >> x >> y >> z;
            scene.mainCamera.position = glm::vec3(x, y, z);
            ls >> x >> y >> z;
            scene.mainCamera.gaze = glm::vec3(x, y, z);
            ls >> scene.mainCamera.fov;
            uint32_t w, h;
            ls >> w >> h;
            scene.mainCamera.resolution = glm::uvec2(w, h);
        } else if(type == "object") {
            Sphere s;
            float x, y, z;
            s.material._padding0 = 0.0F;
            ls >> x >> y >> z;
            s.position = glm::vec3(x, y, z);
            ls >> s.radius;
            ls >> x >> y >> z;
            s.material.color = glm::vec3(x, y, z);
            ls >> x >> y >> z;
            s.material.emission = glm::vec3(x, y, z);
            uint32_t bsdf;
            ls >> bsdf;
            s.material.surfaceType = static_cast<SurfaceType>(bsdf);
            scene.objects.push_back(s);
        } else if (type == "light") {
            int index;
            ls >> index;
            scene.lights.push_back(index);
            scene.lights.front() = scene.lights.size() - 1;
        } else {
            std::printf("unknown type: %s\n", type.c_str());
            std::abort();
        }
    }
}

Scene mainScene;
Config config;

int32_t main(int32_t argc, const char **argv) {
    // Initialize GLFW and create display window.
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(kWindowWidth, kWindowHeight, "GLSL-CS Path Tracer", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));

    // Load scene.
    loadScene("scene/cornell.txt", mainScene);

    // Default config.
    config.maxBounce = 8;
    config.enableRR = false;
    config.rrStartBounce = 4;
    config.spp = 256;

    // Print information.
    int workGroupSizes[3] = {0};
    int workGroupInvocations = 0;
    for (int i = 0; i < 3; ++i)
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &workGroupSizes[i]);
    std::printf("Work group size: (%d, %d, %d)\n", workGroupSizes[0], workGroupSizes[1], workGroupSizes[2]);
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &workGroupInvocations);
    std::printf("Max local work group invocations: %d\n", workGroupInvocations);

    // Compile and link shader programs.
    ComputeShader computeProgram {kCSPath};
    Shader displayProgram {kVSPath, kFSPath};

    // Image binding.
    GLuint imageOut;
    glGenTextures(1, &imageOut);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, imageOut);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, kImageWidth, kImageHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindImageTexture(0, imageOut, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    // Object buffer.
    GLuint sceneObjectsSSB;
    glCreateBuffers(1, &sceneObjectsSSB);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sceneObjectsSSB);
    int32_t objectSize[4] = {static_cast<int>(mainScene.objects.size()), 0, 0, 0};
    glBufferData(GL_SHADER_STORAGE_BUFFER, 
        sizeof(objectSize) + sizeof(Sphere) * mainScene.objects.size(), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(objectSize), objectSize);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(objectSize), 
        sizeof(Sphere) * mainScene.objects.size(), mainScene.objects.data());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sceneObjectsSSB);

    // Light buffer.
    GLuint sceneLightsSSB;
    glCreateBuffers(1, &sceneLightsSSB);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sceneLightsSSB);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 
        sizeof(int) * mainScene.lights.size(), mainScene.lights.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, sceneLightsSSB);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Vertext buffer.
    GLuint vb;
    glCreateBuffers(1, &vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVertices), kQuadVertices, GL_STATIC_DRAW);

    // Index buffer.
    GLuint ib;
    glCreateBuffers(1, &ib);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(kQuadIndices), kQuadIndices, GL_STATIC_DRAW);

    // Vertex array.
    GLuint va;
    glGenVertexArrays(1, &va);
    glBindVertexArray(va);

    // For `aPosition` (vec3).
    glEnableVertexAttribArray(0);
    glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(0, 0);

    // For `aTexCoords` (vec2).
    glEnableVertexAttribArray(1);
    glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 3);
    glVertexAttribBinding(1, 0);

    GLuint timeQuery;
    glGenQueries(1, &timeQuery);
    glBeginQuery(GL_TIME_ELAPSED, timeQuery);

    glUseProgram(computeProgram);
    uniformCamera(computeProgram, mainScene.mainCamera);
    uniformConfig(computeProgram, config);
    glDispatchCompute(kTileWidth, kTileHeight, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glEndQuery(GL_TIME_ELAPSED);

    GLint queryDone = 0;
    while (!queryDone) {
        glGetQueryObjectiv(timeQuery, GL_QUERY_RESULT_AVAILABLE, &queryDone);
    }
    GLint64 elapsed;
    glGetQueryObjecti64v(timeQuery, GL_QUERY_RESULT, &elapsed);
    std::printf("Time cost: %fs\n", elapsed / 1000000000.0);
    glDeleteQueries(1, &timeQuery);

    // Write the result image to a file.
    GLuint fb;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, imageOut, 0);
    float *imageData = new float[kImageWidth * kImageHeight * 4];
    glReadPixels(0, 0, kImageWidth, kImageHeight, GL_RGBA, GL_FLOAT, imageData);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fb);
    stbi_flip_vertically_on_write(true);
    stbi_write_hdr("image.exr", kImageWidth, kImageHeight, 4, imageData);
    delete[] imageData;

    // Main loop.
    while (!glfwWindowShouldClose(window)) {
        // Show the result.
        glUseProgram(displayProgram);
        glBindVertexArray(va);
        glBindVertexBuffer(0, vb, 0, sizeof(float) * 5);
        glVertexArrayElementBuffer(va, ib);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteProgram(displayProgram);
    glDeleteProgram(computeProgram);
    glDeleteVertexArrays(1, &va);
    glDeleteBuffers(1, &ib);
    glDeleteBuffers(1, &vb);
    glDeleteBuffers(1, &sceneLightsSSB);
    glDeleteBuffers(1, &sceneObjectsSSB);
    glDeleteTextures(1, &imageOut);
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}