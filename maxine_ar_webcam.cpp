/*
 * Minimal headless Maxine AR SDK Eye Contact pipeline.
 * Captures from V4L2, processes frame-by-frame with AR SDK, outputs to v4l2loopback.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "nvAR.h"
#include "nvAR_defs.h"
#include "nvARGazeRedirection.h"
#include "gazeEngine.h"

#define BAIL(err, code) \
  do {                  \
    err = code;         \
    goto bail;          \
  } while (0)

int main(int argc, char** argv) {
  const char* camDevice = "/dev/video0";
  const char* outDevice = "/dev/video10";
  int width = 640;
  int height = 480;
  int fps = 30;
  const char* modelPath = "/usr/local/ARSDK/lib/models";

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--cam") == 0 && i + 1 < argc) camDevice = argv[++i];
    else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) outDevice = argv[++i];
    else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) width = atoi(argv[++i]);
    else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) height = atoi(argv[++i]);
    else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) fps = atoi(argv[++i]);
    else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) modelPath = argv[++i];
  }

  // Open camera
  cv::VideoCapture cap(camDevice, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open camera: " << camDevice << std::endl;
    return 1;
  }
  cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
  cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  cap.set(cv::CAP_PROP_FPS, fps);
  for (int i = 0; i < 5; ++i) {
    cv::Mat dummy;
    cap.read(dummy);
  }
  int actual_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int actual_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int actual_fps = (int)cap.get(cv::CAP_PROP_FPS);
  std::cout << "Camera: " << actual_w << "x" << actual_h << " @ " << actual_fps << " fps" << std::endl;

  // Start FFmpeg for v4l2loopback output
  std::string ffmpegCmd = std::string("ffmpeg -hide_banner -loglevel error -y ")
    + "-f rawvideo -pix_fmt bgr24 -s " + std::to_string(width) + "x" + std::to_string(height)
    + " -r " + std::to_string(fps) + " -i - "
    + "-f v4l2 -pix_fmt yuv420p " + outDevice;
  FILE* ffmpegPipe = popen(ffmpegCmd.c_str(), "w");
  if (!ffmpegPipe) {
    std::cerr << "Failed to start FFmpeg for output" << std::endl;
    return 1;
  }
  std::cout << "Output: " << outDevice << std::endl;

  // Initialize AR SDK GazeEngine
  GazeEngine engine;
  engine.setInputImageWidth(width);
  engine.setInputImageHeight(height);
  engine.setGazeRedirect(true);
  engine.setUseCudaGraph(true);
  engine.setEyeSizeSensitivity(3);
  engine.setNumLandmarks(68);
  engine.setFaceStabilization(true);

  GazeEngine::Err err = engine.createGazeRedirectionFeature(modelPath);
  if (err != GazeEngine::Err::errNone) {
    std::cerr << "Failed to create gaze redirection feature" << std::endl;
    pclose(ffmpegPipe);
    return 1;
  }

  err = engine.initGazeRedirectionIOParams();
  if (err != GazeEngine::Err::errNone) {
    std::cerr << "Failed to init gaze redirection IO params" << std::endl;
    engine.destroyGazeRedirectionFeature();
    pclose(ffmpegPipe);
    return 1;
  }

  std::cout << "AR SDK initialized. Running..." << std::endl;

  cv::Mat frame;
  cv::Mat outputFrame(height, width, CV_8UC3);
  int frameCount = 0;
  auto t0 = std::chrono::high_resolution_clock::now();
  double totalProcessMs = 0.0;
  double maxProcessMs = 0.0;

  while (true) {
    if (!cap.read(frame)) continue;
    if (frame.empty()) continue;
    if (frame.cols != width || frame.rows != height) {
      cv::resize(frame, frame, cv::Size(width, height));
    }

    auto t_proc0 = std::chrono::high_resolution_clock::now();
    err = engine.acquireGazeRedirection(frame, outputFrame);
    auto t_proc1 = std::chrono::high_resolution_clock::now();
    double processMs = std::chrono::duration<double, std::milli>(t_proc1 - t_proc0).count();
    totalProcessMs += processMs;
    if (processMs > maxProcessMs) maxProcessMs = processMs;

    if (err == GazeEngine::Err::errNoFace) {
      outputFrame = frame.clone();  // pass through if no face detected
    } else if (err != GazeEngine::Err::errNone) {
      std::cerr << "Gaze redirection error" << std::endl;
      break;
    }

    // Write to FFmpeg pipe
    if (fwrite(outputFrame.data, 1, width * height * 3, ffmpegPipe) != (size_t)(width * height * 3)) {
      std::cerr << "FFmpeg pipe broken" << std::endl;
      break;
    }
    fflush(ffmpegPipe);

    ++frameCount;
    if (frameCount % 300 == 0) {
      auto t1 = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      std::cout << "Processed " << frameCount << " frames in " << elapsed << "s ("
                << (frameCount / elapsed) << " fps) avg=" << (totalProcessMs / frameCount)
                << "ms max=" << maxProcessMs << "ms" << std::endl;
    }
  }

bail:
  std::cout << "Shutting down..." << std::endl;
  engine.destroyGazeRedirectionFeature();
  pclose(ffmpegPipe);
  cap.release();
  return 0;
}
