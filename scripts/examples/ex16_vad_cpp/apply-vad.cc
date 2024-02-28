/* Created on 2017-03-01
 * Author: Binbin Zhang
 * Modified on 2023-06-17 by Sheng Zhao
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>

#include "parse-option.h"
#include "wav.h"
#include "vad.h"

int main(int argc, char *argv[]) {
    const char *usage = "Apply energy vad for input wav file\n"
                        "Usage: vad-test wav_in_file\n";
    ParseOptions po(usage);

    float frame_len = 0.025; // 25 ms
    po.Register("frame-len", &frame_len, "frame length for mvdr");
    float frame_shift = 0.01; // 10ms
    po.Register("frame-shift", &frame_shift, "frame shift for mvdr");
    float energy_thresh = 1.5e7;
    po.Register("energy-thresh", &energy_thresh, 
            "energy threshold for energy based vad");
    int sil_to_speech_trigger = 10;
    po.Register("sil-to-speech-trigger", &sil_to_speech_trigger,
            "num frames for silence to speech trigger");
    int speech_to_sil_trigger = 3;
    po.Register("speech-to-sil-trigger", &speech_to_sil_trigger,
            "num frames for speech to silence trigger");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
        po.PrintUsage();
        exit(1);
    }

    std::string wav_in = po.GetArg(1);
    std::string      wav_out = po.GetArg(2);
    std::string      txt_out = po.GetArg(3);

    WaveReader reader(wav_in.c_str(), 1, 16);

    // printf("input file %s info: \n"
    //        "sample_rate %d \n"
    //        "channels %d \n"
    //        "bits_per_sample_ %d \n",
    //        wav_in.c_str(),
    //        reader.SampleRate(), 
    //        reader.NumChannel(),
    //        reader.BitsPerSample());
    
    int sample_rate = reader.SampleRate();
    int num_sample = reader.NumSample();
    // std::cout<<"num_sample: "<<num_sample<<std::endl;
    int num_point_per_frame = (int)(frame_len * sample_rate);
    int num_point_shift = (int)(frame_shift * sample_rate);
   
    float *data = (float *)calloc(sizeof(float), num_sample);
    // Copy first channel
    for (int i = 0; i < num_sample; i++) {
        data[i] = reader.Data()[i * reader.NumChannel()];
    }

    Vad vad(energy_thresh, sil_to_speech_trigger, speech_to_sil_trigger);

    int num_frames = (num_sample - num_point_per_frame) / num_point_shift + 1;
    std::vector<int> vad_reslut;
    int num_speech_frames = 0;

    // for (int i = 0; i < num_sample; i += num_point_shift) {
    //     // last frame 
    //     if (i + num_point_per_frame > num_sample) break;
    //     int tags = vad.IsSpeech(data+i, num_point_per_frame) ? 1 : 0;
    //     vad_reslut.push_back(tags);
    //     if (tags == 1) num_speech_frames++;
    //     // printf("%f %d \n", float(i) / sample_rate, tags);
    // }

        
    FILE *fp_out = fopen(txt_out.c_str(), "w");

    for (int i = 0; i < num_sample; i += num_point_shift) {
        // last frame 
        if (i + num_point_per_frame > num_sample) break;
        int tags = vad.IsSpeech(data+i, num_point_per_frame) ? 1 : 0;
        vad_reslut.push_back(tags);
        if (tags == 1) num_speech_frames++;
        for (int j = 0; j < num_point_shift; j++)
            fprintf(fp_out, "%d ", tags);
        // printf("%f %d \n", float(i) / sample_rate, tags);
    }

    fclose(fp_out);

    int num_speech_sample = 
             (num_speech_frames - 1) * num_point_shift + num_point_per_frame;
    // std::cout<<"num_speech_sample: "<<num_speech_sample<<std::endl;
    float *speech_data = (float *)calloc(sizeof(float), num_speech_sample);
    
    int speech_cur = 0;
    for (int i = 0; i < vad_reslut.size(); i++) {
        // speech
        if (vad_reslut[i] == 1) {
            memcpy(speech_data + speech_cur * num_point_shift,
                   data + i * num_point_shift, 
                   num_point_per_frame * sizeof(float));
            speech_cur++;
        }
    }

    WaveWriter writer(speech_data, num_speech_sample, reader.NumChannel(),
                     reader.BitDepth());
    writer.Write(wav_out.c_str());
    free(data);
    free(speech_data);
    return 0;
}