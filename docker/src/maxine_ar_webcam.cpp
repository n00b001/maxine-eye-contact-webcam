/*
 * Minimal headless webcam -> gaze redirection -> v4l2loopback bridge
 * Uses NVIDIA AR SDK GazeEngine (from official samples)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include <opencv2/opencv.hpp>
#include "gazeEngine.h"

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int) {
    g_running = 0;
}

static int set_v4l2_format(int fd, int width, int height, uint32_t pixelformat) {
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    fmt.fmt.pix.bytesperline = width * 3;  // BGR24
    fmt.fmt.pix.sizeimage = width * height * 3;
    fmt.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;
    return ioctl(fd, VIDIOC_S_FMT, &fmt);
}

int main(int argc, char** argv) {
    const char* modelPath = getenv("NVAR_MODEL_DIR");
    if (!modelPath) modelPath = "/usr/local/ARSDK/lib/models";

    bool doWarmup = true;
    bool useMjpeg = false;
    int posArgIdx = 0;
    const char* posArgs[3] = {nullptr, nullptr, nullptr};
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--no-warmup") == 0) {
            doWarmup = false;
        } else if (strcmp(argv[i], "--mjpeg") == 0) {
            useMjpeg = true;
        } else if (posArgIdx < 3) {
            posArgs[posArgIdx++] = argv[i];
        }
    }

    const char* v4l2Dev = posArgs[0] ? posArgs[0] : "/dev/video10";
    int width = posArgs[1] ? atoi(posArgs[1]) : 640;
    int height = posArgs[2] ? atoi(posArgs[2]) : 480;

    // Install signal handlers for clean shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Open camera
    const char* camDevice = getenv("CAMERA_DEVICE");
    int camIndex = 0;
    if (camDevice) {
        camIndex = atoi(camDevice);
    }
    cv::VideoCapture cap(camIndex, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open camera index %d\n", camIndex);
        return 1;
    }
    if (useMjpeg) {
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_FPS, 30);
    width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    printf("Camera opened: %dx%d @ %.1f fps\n", width, height, cap.get(cv::CAP_PROP_FPS));

    // Open v4l2loopback output
    int v4l2Fd = open(v4l2Dev, O_WRONLY);
    if (v4l2Fd < 0) {
        fprintf(stderr, "Failed to open %s: %s\n", v4l2Dev, strerror(errno));
        return 1;
    }
    if (set_v4l2_format(v4l2Fd, width, height, V4L2_PIX_FMT_BGR24) < 0) {
        fprintf(stderr, "Failed to set V4L2 format on %s: %s\n", v4l2Dev, strerror(errno));
        close(v4l2Fd);
        return 1;
    }
    printf("V4L2 output opened: %s (%dx%d BGR24)\n", v4l2Dev, width, height);

    // Configure logger
    NvAR_ConfigureLogger(3, "stderr", nullptr, nullptr);

    // Init GazeEngine
    GazeEngine engine;
    engine.setInputImageWidth(width);
    engine.setInputImageHeight(height);
    engine.setGazeRedirect(true);
    engine.setUseCudaGraph(true);
    engine.setFaceStabilization(true);

    GazeEngine::Err err = engine.createGazeRedirectionFeature(modelPath);
    if (err != GazeEngine::errNone) {
        fprintf(stderr, "createGazeRedirectionFeature failed: %d\n", err);
        close(v4l2Fd);
        return 1;
    }
    err = engine.initGazeRedirectionIOParams();
    if (err != GazeEngine::errNone) {
        fprintf(stderr, "initGazeRedirectionIOParams failed: %d\n", err);
        engine.destroyGazeRedirectionFeature();
        close(v4l2Fd);
        return 1;
    }

    printf("Gaze engine initialized. Streaming to %s...\n", v4l2Dev);

    cv::Mat frame, outputFrame;
    const size_t frameSize = width * height * 3;

    // Warmup: run 30 frames through the pipeline without writing to v4l2loopback
    if (doWarmup) {
        const int WARMUP_FRAMES = 30;
        printf("Warming up %d frames...\n", WARMUP_FRAMES);
        for (int i = 0; i < WARMUP_FRAMES && g_running; ++i) {
            cap >> frame;
            if (frame.empty()) break;
            outputFrame.create(height, width, frame.type());
            engine.acquireGazeRedirection(frame, outputFrame);
        }
        printf("Warmup complete.\n");
    }

    int frameCount = 0;
    int latencyCount = 0;
    double latencySum = 0.0;
    double latencyMax = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();

    while (g_running) {
        auto t_frame_start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty()) {
            fprintf(stderr, "Camera frame empty, exiting.\n");
            break;
        }

        outputFrame.create(height, width, frame.type());
        err = engine.acquireGazeRedirection(frame, outputFrame);

        const cv::Mat& out = (err == GazeEngine::errNone) ? outputFrame : frame;

        if (out.isContinuous()) {
            ssize_t written = write(v4l2Fd, out.data, frameSize);
            if (written < 0) {
                fprintf(stderr, "V4L2 write error: %s\n", strerror(errno));
                break;
            }
        } else {
            // Write row by row if not continuous
            for (int y = 0; y < out.rows; ++y) {
                ssize_t written = write(v4l2Fd, out.ptr(y), out.cols * out.elemSize());
                if (written < 0) {
                    fprintf(stderr, "V4L2 write error: %s\n", strerror(errno));
                    break;
                }
            }
        }

        auto t_frame_end = std::chrono::high_resolution_clock::now();
        double latencyMs = std::chrono::duration<double, std::milli>(t_frame_end - t_frame_start).count();
        latencySum += latencyMs;
        latencyMax = std::max(latencyMax, latencyMs);
        ++latencyCount;

        ++frameCount;
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        if (elapsed >= 5.0) {
            printf("FPS: %.1f  Latency avg: %.2f ms  max: %.2f ms\n",
                   frameCount / elapsed,
                   latencySum / latencyCount,
                   latencyMax);
            fflush(stdout);
            frameCount = 0;
            latencySum = 0.0;
            latencyMax = 0.0;
            latencyCount = 0;
            t0 = t1;
        }
    }

    close(v4l2Fd);
    cap.release();
    engine.destroyGazeRedirectionFeature();
    return 0;
}
