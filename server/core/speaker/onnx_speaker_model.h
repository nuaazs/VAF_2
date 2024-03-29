#ifndef SPEAKER_ONNX_SPEAKER_MODEL_H_
#define SPEAKER_ONNX_SPEAKER_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "speaker/speaker_model.h"

namespace wespeaker {

class OnnxSpeakerModel : public SpeakerModel {
 public:
  static void InitEngineThreads(int num_threads = 1);
#ifdef USE_GPU
  static void SetGpuDeviceId(int gpu_id = 0);
#endif
 public:
  explicit OnnxSpeakerModel();

  void ExtractEmbedding(const std::vector<std::vector<float>>& feats,
                        std::vector<float>* embed) override;

 private:
  // session
  static Ort::Env env_;
  static Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::Session> speaker_session_ = nullptr;
  // node names
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  int embedding_size_ = 0;
};

}  // namespace wespeaker

#endif  // SPEAKER_ONNX_SPEAKER_MODEL_H_