// SPDX-License-Identifier: MIT
//
// Zero-copy pybind11 shim over NVIDIA Maxine AR SDK GazeRedirection.
//
// Input/output frames are plain CUDA device pointers owned by the Python
// caller; they are wrapped into NvCVImage descriptors per-frame via
// NvCVImage_Init (never NvCVImage_Alloc, never NvCVImage_Transfer). The
// caller supplies the CUDA stream; when 0 is passed we create one via
// NvAR_CudaStreamCreate and destroy it ourselves on teardown.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nvAR.h"
#include "nvAR_defs.h"
#include "nvARGazeRedirection.h"
#include "nvCVImage.h"
#include "nvCVStatus.h"

namespace py = pybind11;

namespace {

inline void check(NvCV_Status s, const char* what) {
  if (s != NVCV_SUCCESS) {
    std::string msg = std::string("AR SDK ") + what + ": " +
                      NvCV_GetErrorStringFromCode(s);
    throw std::runtime_error(msg);
  }
}

// num_output_landmarks is the size of the GazeOutputLandmarks buffer
// produced by the gaze-redirection network. The SDK default (per the
// gazeEngine.cpp reference) is 12 eye-region landmarks per frame.
constexpr int kNumGazeOutputLandmarks = 12;

// Landmark confidence threshold below which run() reports no-face.
// Mirrors the LANDMARKS_INFO entries in gazeEngine.h for 68 and 126 points.
constexpr float kConfidenceThreshold = 0.15f;

// Output gaze angle vector size: (pitch, yaw).
constexpr unsigned int kOutputGazeSize = 2;
// Head translation size: (x, y, z).
constexpr unsigned int kHeadTranslationSize = 3;

class GazeRedirect {
 public:
  GazeRedirect(unsigned int width,
               unsigned int height,
               const std::string& model_dir,
               std::uintptr_t cuda_stream_ptr,
               unsigned int num_landmarks,
               bool use_cuda_graph,
               bool stabilize,
               bool gaze_redirect,
               unsigned int eye_size_sensitivity,
               float gaze_pitch_threshold_low,
               float gaze_yaw_threshold_low,
               float head_pitch_threshold_low,
               float head_yaw_threshold_low,
               float gaze_pitch_threshold_high,
               float gaze_yaw_threshold_high,
               float head_pitch_threshold_high,
               float head_yaw_threshold_high)
      : width_(width),
        height_(height),
        num_landmarks_(num_landmarks),
        batch_size_(1) {
    if (num_landmarks != 68 && num_landmarks != 126) {
      throw std::runtime_error(
          "num_landmarks must be 68 or 126 (AR SDK GazeRedirection)");
    }

    if (cuda_stream_ptr == 0) {
      check(NvAR_CudaStreamCreate(&stream_), "CudaStreamCreate");
      owns_stream_ = true;
    } else {
      stream_ = reinterpret_cast<CUstream>(cuda_stream_ptr);
      owns_stream_ = false;
    }

    // Mirrors gazeEngine.cpp:91-131 ordering exactly.
    check(NvAR_Create(NvAR_Feature_GazeRedirection, &handle_), "Create");
    check(NvAR_SetString(handle_, NvAR_Parameter_Config(ModelDir),
                         model_dir.c_str()),
          "SetString(ModelDir)");
    check(NvAR_SetU32(handle_, NvAR_Parameter_Config(Landmarks_Size),
                      num_landmarks_),
          "SetU32(Landmarks_Size)");
    // -1 = 0xffffffff turns on all temporal filtering (matches gazeEngine.cpp).
    check(NvAR_SetU32(handle_, NvAR_Parameter_Config(Temporal),
                      stabilize ? static_cast<unsigned int>(-1) : 0u),
          "SetU32(Temporal)");
    check(NvAR_SetU32(handle_, NvAR_Parameter_Config(GazeRedirect),
                      gaze_redirect ? 1u : 0u),
          "SetU32(GazeRedirect)");
    check(NvAR_SetCudaStream(handle_, NvAR_Parameter_Config(CUDAStream),
                             stream_),
          "SetCudaStream");
    check(NvAR_SetU32(handle_, NvAR_Parameter_Config(UseCudaGraph),
                      use_cuda_graph ? 1u : 0u),
          "SetU32(UseCudaGraph)");
    check(NvAR_SetU32(handle_, NvAR_Parameter_Config(EyeSizeSensitivity),
                      eye_size_sensitivity),
          "SetU32(EyeSizeSensitivity)");
    check(NvAR_SetF32(handle_, NvAR_Parameter_Config(GazePitchThresholdLow),
                      gaze_pitch_threshold_low),
          "SetF32(GazePitchThresholdLow)");
    check(NvAR_SetF32(handle_, NvAR_Parameter_Config(GazeYawThresholdLow),
                      gaze_yaw_threshold_low),
          "SetF32(GazeYawThresholdLow)");
    check(NvAR_SetF32(handle_, NvAR_Parameter_Config(HeadPitchThresholdLow),
                      head_pitch_threshold_low),
          "SetF32(HeadPitchThresholdLow)");
    check(NvAR_SetF32(handle_, NvAR_Parameter_Config(HeadYawThresholdLow),
                      head_yaw_threshold_low),
          "SetF32(HeadYawThresholdLow)");
    check(NvAR_SetF32(handle_, NvAR_Parameter_Config(GazePitchThresholdHigh),
                      gaze_pitch_threshold_high),
          "SetF32(GazePitchThresholdHigh)");
    check(NvAR_SetF32(handle_, NvAR_Parameter_Config(GazeYawThresholdHigh),
                      gaze_yaw_threshold_high),
          "SetF32(GazeYawThresholdHigh)");
    check(NvAR_SetF32(handle_, NvAR_Parameter_Config(HeadPitchThresholdHigh),
                      head_pitch_threshold_high),
          "SetF32(HeadPitchThresholdHigh)");
    check(NvAR_SetF32(handle_, NvAR_Parameter_Config(HeadYawThresholdHigh),
                      head_yaw_threshold_high),
          "SetF32(HeadYawThresholdHigh)");
    check(NvAR_Load(handle_), "Load");

    gaze_redirect_ = gaze_redirect;
    init_io();
  }

  GazeRedirect(const GazeRedirect&) = delete;
  GazeRedirect& operator=(const GazeRedirect&) = delete;

  ~GazeRedirect() {
    if (handle_) {
      (void)NvAR_Destroy(handle_);
      handle_ = nullptr;
    }
    if (stream_ && owns_stream_) {
      (void)NvAR_CudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  // run() binds caller-supplied CUDA device pointers + pitches into the
  // input/output NvCVImage descriptors, re-registers them with the SDK, and
  // executes the feature. Returns false only when average landmark
  // confidence is below threshold (SDK saw no face); other SDK errors raise.
  bool run(std::uintptr_t in_dptr,
           unsigned int in_pitch,
           std::uintptr_t out_dptr,
           unsigned int out_pitch) {
    if (in_dptr == 0) {
      throw std::runtime_error("run(): in_dptr is null");
    }
    if (gaze_redirect_ && out_dptr == 0) {
      throw std::runtime_error("run(): out_dptr is null (gaze_redirect=True)");
    }

    // Input descriptor — zero-copy view into caller GPU buffer.
    check(NvCVImage_Init(&in_img_, width_, height_, static_cast<int>(in_pitch),
                         reinterpret_cast<void*>(in_dptr), NVCV_BGR, NVCV_U8,
                         NVCV_CHUNKY, NVCV_GPU),
          "NvCVImage_Init(input)");
    in_img_.deletePtr = nullptr;
    in_img_.deleteProc = nullptr;
    check(NvAR_SetObject(handle_, NvAR_Parameter_Input(Image), &in_img_,
                         sizeof(NvCVImage)),
          "SetObject(Input.Image)");

    if (gaze_redirect_) {
      check(NvCVImage_Init(&out_img_, width_, height_,
                           static_cast<int>(out_pitch),
                           reinterpret_cast<void*>(out_dptr), NVCV_BGR, NVCV_U8,
                           NVCV_CHUNKY, NVCV_GPU),
            "NvCVImage_Init(output)");
      out_img_.deletePtr = nullptr;
      out_img_.deleteProc = nullptr;
      check(NvAR_SetObject(handle_, NvAR_Parameter_Output(Image), &out_img_,
                           sizeof(NvCVImage)),
            "SetObject(Output.Image)");
    }

    check(NvAR_Run(handle_), "Run");

    return average_landmark_confidence() >= kConfidenceThreshold;
  }

  float face_confidence() const {
    return const_cast<GazeRedirect*>(this)->average_landmark_confidence();
  }

  std::pair<float, float> gaze_vector() const {
    return {gaze_angles_vector_[0], gaze_angles_vector_[1]};
  }

  unsigned int num_landmarks() const { return num_landmarks_; }

 private:
  // Wire up the host-resident output buffers and input dimension scalars.
  // Image input/output are bound per-frame in run(), not here.
  void init_io() {
    check(NvAR_SetS32(handle_, NvAR_Parameter_Input(Width),
                      static_cast<int>(width_)),
          "SetS32(Input.Width)");
    check(NvAR_SetS32(handle_, NvAR_Parameter_Input(Height),
                      static_cast<int>(height_)),
          "SetS32(Input.Height)");

    unsigned int kpts_size = 0;
    check(NvAR_GetU32(handle_, NvAR_Parameter_Config(Landmarks_Size),
                      &kpts_size),
          "GetU32(Landmarks_Size)");

    facial_landmarks_.assign(batch_size_ * kpts_size, {0.f, 0.f});
    check(NvAR_SetObject(handle_, NvAR_Parameter_Output(Landmarks),
                         facial_landmarks_.data(), sizeof(NvAR_Point2f)),
          "SetObject(Output.Landmarks)");

    gaze_output_landmarks_.assign(batch_size_ * kNumGazeOutputLandmarks,
                                  {0.f, 0.f});
    check(NvAR_SetObject(handle_, NvAR_Parameter_Output(GazeOutputLandmarks),
                         gaze_output_landmarks_.data(), sizeof(NvAR_Point2f)),
          "SetObject(Output.GazeOutputLandmarks)");

    facial_landmarks_confidence_.assign(batch_size_ * kpts_size, 0.f);
    check(NvAR_SetF32Array(
              handle_, NvAR_Parameter_Output(LandmarksConfidence),
              facial_landmarks_confidence_.data(),
              static_cast<int>(batch_size_ * kpts_size)),
          "SetF32Array(Output.LandmarksConfidence)");

    check(NvAR_SetF32Array(
              handle_, NvAR_Parameter_Output(OutputGazeVector),
              gaze_angles_vector_,
              static_cast<int>(batch_size_ * kOutputGazeSize)),
          "SetF32Array(Output.OutputGazeVector)");

    check(NvAR_SetF32Array(
              handle_, NvAR_Parameter_Output(OutputHeadTranslation),
              head_translation_,
              static_cast<int>(batch_size_ * kHeadTranslationSize)),
          "SetF32Array(Output.OutputHeadTranslation)");

    check(NvAR_SetObject(handle_, NvAR_Parameter_Output(HeadPose),
                         &head_pose_, sizeof(NvAR_Quaternion)),
          "SetObject(Output.HeadPose)");

    check(NvAR_SetObject(handle_, NvAR_Parameter_Output(GazeDirection),
                         gaze_direction_, sizeof(NvAR_Point3f)),
          "SetObject(Output.GazeDirection)");

    output_bbox_data_.assign(batch_size_, {0.f, 0.f, 0.f, 0.f});
    output_bboxes_.boxes = output_bbox_data_.data();
    output_bboxes_.max_boxes = static_cast<uint8_t>(batch_size_);
    output_bboxes_.num_boxes = static_cast<uint8_t>(batch_size_);
    check(NvAR_SetObject(handle_, NvAR_Parameter_Output(BoundingBoxes),
                         &output_bboxes_, sizeof(NvAR_BBoxes)),
          "SetObject(Output.BoundingBoxes)");

    num_kpts_total_ = batch_size_ * kpts_size;
  }

  float average_landmark_confidence() {
    if (num_kpts_total_ == 0 || facial_landmarks_confidence_.empty()) {
      return 0.f;
    }
    float sum = 0.f;
    for (unsigned int i = 0; i < num_kpts_total_; ++i) {
      sum += facial_landmarks_confidence_[i];
    }
    return sum / static_cast<float>(num_kpts_total_);
  }

  unsigned int width_;
  unsigned int height_;
  unsigned int num_landmarks_;
  unsigned int batch_size_;
  bool gaze_redirect_ = true;
  bool owns_stream_ = false;
  unsigned int num_kpts_total_ = 0;

  NvAR_FeatureHandle handle_ = nullptr;
  CUstream stream_ = nullptr;

  // Per-frame NvCVImage descriptors. Re-populated on every run(); they
  // never own their pixel buffers (deletePtr/deleteProc stay null).
  NvCVImage in_img_{};
  NvCVImage out_img_{};

  // Host-resident SDK outputs. The SDK expects these buffers to be CPU-side
  // and small (<1KB total), so there is no zero-copy win in moving them to
  // the GPU.
  std::vector<NvAR_Point2f> facial_landmarks_;
  std::vector<NvAR_Point2f> gaze_output_landmarks_;
  std::vector<float> facial_landmarks_confidence_;
  NvAR_Point3f gaze_direction_[2] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
  NvAR_Quaternion head_pose_ = {0.f, 0.f, 0.f, 1.f};
  float gaze_angles_vector_[2] = {0.f, 0.f};
  float head_translation_[3] = {0.f, 0.f, 0.f};
  std::vector<NvAR_Rect> output_bbox_data_;
  NvAR_BBoxes output_bboxes_{};
};

}  // namespace

PYBIND11_MODULE(maxine_ar_ext, m) {
  m.doc() =
      "Zero-copy pybind11 shim over NVIDIA Maxine AR SDK GazeRedirection. "
      "Caller owns CUDA stream and device buffers; NvCVImage descriptors "
      "are re-initialized per-frame against caller pointers.";

  py::class_<GazeRedirect>(m, "GazeRedirect")
      .def(py::init<unsigned int, unsigned int, const std::string&,
                    std::uintptr_t, unsigned int, bool, bool, bool,
                    unsigned int, float, float, float, float, float, float,
                    float, float>(),
           py::arg("width"),
           py::arg("height"),
           py::arg("model_dir"),
           py::arg("cuda_stream_ptr") = static_cast<std::uintptr_t>(0),
           py::arg("num_landmarks") = 126u,
           py::arg("use_cuda_graph") = true,
           py::arg("stabilize") = true,
           py::arg("gaze_redirect") = true,
           py::arg("eye_size_sensitivity") = 3u,
           py::arg("gaze_pitch_threshold_low") = 20.0f,
           py::arg("gaze_yaw_threshold_low") = 20.0f,
           py::arg("head_pitch_threshold_low") = 15.0f,
           py::arg("head_yaw_threshold_low") = 25.0f,
           py::arg("gaze_pitch_threshold_high") = 30.0f,
           py::arg("gaze_yaw_threshold_high") = 30.0f,
           py::arg("head_pitch_threshold_high") = 25.0f,
           py::arg("head_yaw_threshold_high") = 35.0f)
      .def("run", &GazeRedirect::run,
           py::arg("in_dptr"),
           py::arg("in_pitch"),
           py::arg("out_dptr"),
           py::arg("out_pitch"),
           "Run GazeRedirection on the caller's GPU buffers. Returns True "
           "on success, False when average landmark confidence falls below "
           "the no-face threshold (no exception is raised in that case).")
      .def_property_readonly("face_confidence",
                             &GazeRedirect::face_confidence)
      .def_property_readonly("gaze_vector", &GazeRedirect::gaze_vector)
      .def_property_readonly("num_landmarks", &GazeRedirect::num_landmarks);
}
