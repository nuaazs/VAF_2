#ifndef SPEAKER_SPEAKER_MODEL_H_
#define SPEAKER_SPEAKER_MODEL_H_

#include <vector>
#include <string>
#include "utils/utils.h"

namespace wespeaker {

class SpeakerModel {
 public:
  virtual ~SpeakerModel() = default;
  // extract embedding
  // NOTE: https://www.cnblogs.com/zhmlzhml/p/12973618.html
  virtual void ExtractEmbedding(const std::vector<std::vector<float>>& feats,
                                std::vector<float>* embed) {}
};

}  // namespace wespeaker

#endif  // SPEAKER_SPEAKER_MODEL_H_