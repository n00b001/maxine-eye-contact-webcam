#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "nvAR.h"
#include "gazeEngine.h"

int main(int argc, char** argv) {
  const char* modelPath = "/home/alex/Downloads/arsdk-docker/ARSDK/lib/models";
  int width = 640, height = 480;

  GazeEngine engine;
  engine.setInputImageWidth(width);
  engine.setInputImageHeight(height);
  engine.setGazeRedirect(true);
  engine.setUseCudaGraph(true);
  engine.setEyeSizeSensitivity(3);
  engine.setNumLandmarks(68);
  engine.setFaceStabilization(true);

  auto err = engine.createGazeRedirectionFeature(modelPath);
  if (err != GazeEngine::Err::errNone) {
    std::cerr << "createGazeRedirectionFeature failed" << std::endl;
    return 1;
  }
  err = engine.initGazeRedirectionIOParams();
  if (err != GazeEngine::Err::errNone) {
    std::cerr << "initGazeRedirectionIOParams failed" << std::endl;
    return 1;
  }
  std::cout << "Init OK" << std::endl;

  cv::Mat frame = cv::Mat::zeros(height, width, CV_8UC3);
  cv::Mat outputFrame(height, width, CV_8UC3);

  std::cout << "frame: " << frame.rows << "x" << frame.cols << " channels=" << frame.channels()
            << " type=" << frame.type() << " CV_8UC3=" << CV_8UC3 << " continuous=" << frame.isContinuous() << std::endl;
  std::cout << "outputFrame: " << outputFrame.rows << "x" << outputFrame.cols << " channels=" << outputFrame.channels()
            << " type=" << outputFrame.type() << " continuous=" << outputFrame.isContinuous() << std::endl;

  try {
    err = engine.acquireGazeRedirection(frame, outputFrame);
    std::cout << "acquireGazeRedirection returned " << (int)err << std::endl;
  } catch (const cv::Exception& e) {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Std exception: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Done" << std::endl;
  engine.destroyGazeRedirectionFeature();
  return 0;
}
