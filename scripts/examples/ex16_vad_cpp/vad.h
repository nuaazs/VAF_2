/* Created on 2016-08-24
 * Author: Binbin Zhang
 */

#ifndef VAD_H_
#define VAD_H_

#include <assert.h>

typedef enum {
    kSpeech,
    kSilence
} VadState;

class Vad {
public:
    Vad(float energy_thresh, int silence_to_speech_thresh, 
        int speech_to_sil_thresh): 
        energy_thresh_(energy_thresh), 
        silence_to_speech_thresh_(silence_to_speech_thresh),
        speech_to_sil_thresh_(speech_to_sil_thresh),
        silence_frame_count_(0), speech_frame_count_(0), 
        frame_count_(0), state_(kSilence) {
    }

    Vad() {
        Reset();
    }

    void Reset() {
        silence_frame_count_ = 0; 
        speech_frame_count_ = 0;
        frame_count_ = 0; 
        state_ = kSilence; 
    }

    // return 1 if current frame is speech
    bool IsSpeech(float *data, int num_point) {
        float energy = 0.0; 
        bool is_voice = false;
        for (int i = 0; i < num_point; i++) {
            energy += data[i] * data[i];
        }
        if (energy > energy_thresh_) is_voice = true;
        switch (state_) {
            case kSilence:
                if (is_voice) {
                    speech_frame_count_++;
                    if (speech_frame_count_ >= silence_to_speech_thresh_) {
                        state_ = kSpeech;
                        silence_frame_count_ = 0;
                    }
                } else {
                    speech_frame_count_ = 0;
                }
                break;
            case kSpeech:
                if (!is_voice) {
                    silence_frame_count_++;
                    if (silence_frame_count_ >= speech_to_sil_thresh_) {
                        state_ = kSilence;
                        speech_frame_count_ = 0;
                    }
                } else {
                    silence_frame_count_ = 0;
                }
                break;
            default:
                assert(0);
        }
        if (state_ == kSpeech) return true;
        else return false;
    }
private:
    float energy_thresh_;        
    int silence_to_speech_thresh_;
    int speech_to_sil_thresh_;
    int silence_frame_count_;
    int speech_frame_count_;
    int frame_count_;
    VadState state_;
};


#endif



