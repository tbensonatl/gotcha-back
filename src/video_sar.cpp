#include <cstring>
#include <filesystem>
#include <mutex>

#include <cuda_runtime.h>

// clang-format off
// The glew header must be processed before gl.h and glext.h
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/freeglut.h>
// clang-format on

#include <cuda_gl_interop.h>

#include "data_reader.h"
#include "helpers.h"
#include "kernels.h"

namespace fs = std::filesystem;

static struct {
    Gotcha::DataBlock data_set;

    GLuint pbo{0};
    GLuint tex{0};
    struct cudaGraphicsResource *cuda_pbo_resource{nullptr};

    // Device buffers
    cuComplex *dev_range_profiles{nullptr};
    cuComplex *dev_image{nullptr};
    float3 *dev_ant_pos{nullptr};
    float *dev_max_magnitude_workbuf{nullptr};
    uint8_t *dev_bp_workbuf{nullptr};

    // Host pinned buffers
    cuComplex *phase_history{nullptr};
    cuComplex *image{nullptr};
    cuComplex *ant_pos{nullptr};

    cudaStream_t stream{0};
    cufftHandle fft_plan{0};

    VideoParams params;

    int num_upsampled_bins{0};
    int start_az_next_image{0};

    // The lock is only needed for user-adjustable parameters like
    // the number of azimuth degrees per image.
    std::mutex m_lock;

    int adjustAzimuthDegreesPerImage(int adj) {
        std::lock_guard<std::mutex> guard(m_lock);
        params.az_degrees_per_image =
            std::max(0, std::min(params.az_degrees_per_image + adj, 360));
        return params.az_degrees_per_image;
    }

    int getAzimuthDegreesPerImage() {
        std::lock_guard<std::mutex> guard(m_lock);
        return params.az_degrees_per_image;
    }
} s_state;

static void initCuda() {
    const Gotcha::DataRequest request = {
        .first_az = 1,
        .last_az = 360,
        .pass = s_state.params.pass,
        .polarization = Gotcha::Polarization::HH,
    };

    Gotcha::Reader reader(s_state.params.gotcha_dir);
    const SarReturnCode rc = reader.ReadDataSet(s_state.data_set, request);
    if (rc != SarReturnCode::Success) {
        LOG_ERR("Failed to read data set: %s", GetSarReturnCodeString(rc));
        exit(EXIT_FAILURE);
    }

    s_state.num_upsampled_bins =
        static_cast<int>(pow(2, ceil(log(s_state.params.range_upsample_factor *
                                         s_state.data_set.num_freq) /
                                     log(2))));

    const int n_pulses = s_state.data_set.num_pulses;
    const int n_freq = s_state.data_set.num_freq;
    const int nx = s_state.params.image_width;
    const int ny = s_state.params.image_height;

    LOG("Loaded %d pulses", n_pulses);

    cudaChecked(
        cudaMalloc((void **)&s_state.dev_range_profiles,
                   n_pulses * s_state.num_upsampled_bins * sizeof(cuComplex)));
    cudaChecked(
        cudaMalloc((void **)&s_state.dev_image, nx * ny * sizeof(cuComplex)));
    cudaChecked(
        cudaMalloc((void **)&s_state.dev_ant_pos, sizeof(float3) * n_pulses));
    const size_t mag_workbuf_size_bytes = GetMaxMagnitudeWorkBufSizeBytes(ny);
    cudaChecked(cudaMalloc((void **)&s_state.dev_max_magnitude_workbuf,
                           mag_workbuf_size_bytes));

    const size_t bp_workbuf_bytes = GetBackprojWorkBufSizeBytes(
        s_state.params.kernel, s_state.num_upsampled_bins);
    if (bp_workbuf_bytes > 0) {
        cudaChecked(
            cudaMalloc((void **)&s_state.dev_bp_workbuf, bp_workbuf_bytes));
    }

    cudaChecked(cudaMallocHost((void **)&s_state.phase_history,
                               n_pulses * n_freq * sizeof(cuComplex)));
    cudaChecked(
        cudaMallocHost((void **)&s_state.image, nx * ny * sizeof(cuComplex)));
    cudaChecked(cudaMallocHost((void **)&s_state.ant_pos,
                               s_state.data_set.num_pulses * sizeof(float3)));

    memcpy(s_state.phase_history, &s_state.data_set.phase_history[0],
           sizeof(cuComplex) * n_pulses * n_freq);
    memcpy(s_state.ant_pos, &s_state.data_set.ant_pos[0],
           sizeof(float3) * n_pulses);

    cudaChecked(cudaStreamCreate(&s_state.stream));
    cufftChecked(cufftPlan1d(&s_state.fft_plan, s_state.num_upsampled_bins,
                             CUFFT_C2C, n_pulses));
    cufftChecked(cufftSetStream(s_state.fft_plan, s_state.stream));
}

static int isGLVersionSupported(unsigned reqMajor, unsigned reqMinor) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    if (glewInit() != GLEW_OK) {
        std::cerr << "glewInit() failed!" << std::endl;
        return 0;
    }
#endif
    const auto gl_version = glGetString(GL_VERSION);
    LOG("gl_version='%s'", (const char *)gl_version);
    std::string version((const char *)gl_version);
    std::stringstream stream(version);
    unsigned major, minor;
    char dot;

    stream >> major >> dot >> minor;

    return major > reqMajor || (major == reqMajor && minor >= reqMinor);
}

static void cleanup() {
    if (s_state.pbo) {
        cudaGraphicsUnregisterResource(s_state.cuda_pbo_resource);
        glDeleteBuffersARB(1, &s_state.pbo);
        s_state.pbo = 0;
        glDeleteTextures(1, &s_state.tex);
        s_state.tex = 0;
    }

    cudaStreamDestroy(s_state.stream);
    cufftDestroy(s_state.fft_plan);
    cudaFreeHost(s_state.ant_pos);
    cudaFreeHost(s_state.image);
    cudaFreeHost(s_state.phase_history);
    cudaFree(s_state.dev_max_magnitude_workbuf);
    cudaFree(s_state.dev_ant_pos);
    if (s_state.dev_bp_workbuf) {
        cudaFree(s_state.dev_bp_workbuf);
    }
    cudaFree(s_state.dev_image);
    cudaFree(s_state.dev_range_profiles);
    s_state.data_set.Reset();
}

static void render() {
    static bool have_copied_data = false;

    const int n_pulses = s_state.data_set.num_pulses;
    const int n_freq = s_state.data_set.num_freq;
    const int az_degrees_per_image = s_state.getAzimuthDegreesPerImage();

    const int n_pulses_per_az_deg = static_cast<int>(roundf(n_pulses / 360.0f));
    const int first_pulse_this_img =
        s_state.start_az_next_image * n_pulses_per_az_deg;
    const int n_pulses_this_img = n_pulses_per_az_deg * az_degrees_per_image;

    const int nx = s_state.params.image_width;
    const int ny = s_state.params.image_height;

    // map PBO to get CUDA device pointer
    uint *dev_output;
    cudaChecked(cudaGraphicsMapResources(1, &s_state.cuda_pbo_resource, 0));
    size_t num_bytes;
    cudaChecked(cudaGraphicsResourceGetMappedPointer(
        (void **)&dev_output, &num_bytes, s_state.cuda_pbo_resource));

    if (!have_copied_data) {
        cudaChecked(cudaMemcpyAsync(s_state.dev_ant_pos, s_state.ant_pos,
                                    sizeof(float3) * n_pulses,
                                    cudaMemcpyHostToDevice, s_state.stream));
        cudaChecked(cudaMemcpy2DAsync(
            s_state.dev_range_profiles,
            sizeof(cufftComplex) * s_state.num_upsampled_bins,
            s_state.phase_history, sizeof(cufftComplex) * n_freq,
            sizeof(cufftComplex) * n_freq, n_pulses, cudaMemcpyHostToDevice,
            s_state.stream));
        cudaChecked(cudaMemset2DAsync(
            s_state.dev_range_profiles + n_freq,
            sizeof(cuComplex) * s_state.num_upsampled_bins, 0,
            sizeof(cuComplex) * (s_state.num_upsampled_bins - n_freq), n_pulses,
            s_state.stream));
        cufftChecked(cufftExecC2C(s_state.fft_plan, s_state.dev_range_profiles,
                                  s_state.dev_range_profiles, CUFFT_INVERSE));
        FftShiftGpu(s_state.dev_range_profiles, s_state.num_upsampled_bins,
                    n_pulses, s_state.stream);
        have_copied_data = true;
    }

    cudaChecked(cudaMemsetAsync(s_state.dev_image, 0,
                                sizeof(cuComplex) * nx * ny, s_state.stream));

    const double c = SPEED_OF_LIGHT_M_PER_SEC;
    const float maxWr =
        static_cast<float>(c / (2.0 * s_state.data_set.del_freq));
    const float image_field_of_view_m = 150.0f;
    const float dx = image_field_of_view_m / nx;
    const float dy = image_field_of_view_m / ny;
    const float dR = maxWr / s_state.num_upsampled_bins;

    if (first_pulse_this_img + n_pulses_this_img <= n_pulses) {
        SarBpGpu(s_state.dev_image, nx, ny,
                 s_state.dev_range_profiles +
                     first_pulse_this_img * s_state.num_upsampled_bins,
                 s_state.dev_bp_workbuf, s_state.num_upsampled_bins,
                 n_pulses_this_img, s_state.dev_ant_pos + first_pulse_this_img,
                 s_state.data_set.min_freq, dR, dx, dy, 0.0f,
                 s_state.params.kernel, s_state.stream);
    } else {
        SarBpGpu(s_state.dev_image, nx, ny,
                 s_state.dev_range_profiles +
                     first_pulse_this_img * s_state.num_upsampled_bins,
                 s_state.dev_bp_workbuf, s_state.num_upsampled_bins,
                 n_pulses - first_pulse_this_img,
                 s_state.dev_ant_pos + first_pulse_this_img,
                 s_state.data_set.min_freq, dR, dx, dy, 0.0f,
                 s_state.params.kernel, s_state.stream);
        const int pulses_remaining =
            n_pulses_this_img - (n_pulses - first_pulse_this_img);
        SarBpGpu(s_state.dev_image, nx, ny, s_state.dev_range_profiles,
                 s_state.dev_bp_workbuf, s_state.num_upsampled_bins,
                 pulses_remaining, s_state.dev_ant_pos,
                 s_state.data_set.min_freq, dR, dx, dy, 0.0f,
                 s_state.params.kernel, s_state.stream);
    }

    const float min_normalized_db = -70;
    ComputeMagnitudeImage(dev_output, s_state.dev_max_magnitude_workbuf,
                          s_state.dev_image, nx, ny, min_normalized_db,
                          s_state.stream);

    cudaChecked(cudaStreamSynchronize(s_state.stream));

    cudaChecked(cudaGraphicsUnmapResources(1, &s_state.cuda_pbo_resource, 0));

    cudaChecked(cudaDeviceSynchronize());

    s_state.start_az_next_image =
        (s_state.start_az_next_image + az_degrees_per_image) % 360;
}

static void display() {
    render();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, s_state.pbo);
    glBindTexture(GL_TEXTURE_2D, s_state.tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s_state.params.image_width,
                    s_state.params.image_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutReportErrors();
}

static void idle() { glutPostRedisplay(); }

static void keyboard(unsigned char key, int, int) {
    switch (key) {
    case '-': {
        const int az = s_state.adjustAzimuthDegreesPerImage(-5);
        LOG("Using %d azimuth degrees per image\n", az);
    } break;
    case '+': {
        const int az = s_state.adjustAzimuthDegreesPerImage(5);
        LOG("Using %d azimuth degrees per image\n", az);
    } break;
    case 27:
        // Escape key
        exit(EXIT_SUCCESS);
        break;
    }
    glutPostRedisplay();
}

static void mouse(int, int, int, int) { glutPostRedisplay(); }

static void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glutPostRedisplay();
}

void runVideo(const VideoParams &params, int argc, char **argv, int device_id) {
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    int num_devices = -1;
    cudaChecked(cudaGetDeviceCount(&num_devices));
    if (device_id < 0 || device_id >= num_devices) {
        LOG_ERR("Invalid GPU device ID %d; have %d available devices.",
                device_id, num_devices);
        exit(EXIT_FAILURE);
    }

    cudaChecked(cudaSetDevice(device_id));

    s_state.params = params;
    const int window_width = 768;
    const int window_height = 768;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Video SAR");
    glewInit();

    if (!isGLVersionSupported(2, 0)) {
        LOG("Missing necessary GL support");
        exit(EXIT_FAILURE);
    }

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutCloseFunc(cleanup);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, window_width, window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    initCuda();

    const int nx = s_state.params.image_width;
    const int ny = s_state.params.image_height;

    // create pixel buffer object for display
    glGenBuffersARB(1, &s_state.pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, s_state.pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, nx * ny * sizeof(GLubyte) * 4,
                    0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    cudaChecked(cudaGraphicsGLRegisterBuffer(&s_state.cuda_pbo_resource,
                                             s_state.pbo,
                                             cudaGraphicsMapFlagsWriteDiscard));

    glGenTextures(1, &s_state.tex);
    glBindTexture(GL_TEXTURE_2D, s_state.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nx, ny, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutMainLoop();

    const GLenum gl_err = glGetError();
    if (gl_err != GL_NO_ERROR) {
        LOG_ERR("GL error %d", gl_err);
    }
}
