/* Created on 2017-03-01
 * Author: Sheng Zhao
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include <algorithm>

#include "../include_vad/parse-option.h"
#include "../include_vad/wav.h"
#include "../include_vad/vad.h"

int main(int argc, char *argv[]) {
    const char *usage = "Apply energy vad for input wav file\n"
                        "Usage: vad --wav-bin=<wav_in> --text-out=<txt_out> --energy-thresh=<energy-thresh> --smooth-threshold=<smooth_threshold> --min-duration=<min_duration>\n";
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

    std::string wav_bin = "wav_in.bin";
    po.Register("wav-bin", &wav_bin, "input wav file");
    std::string text_out = "text_out.txt";
    po.Register("text-out", &text_out, "output text file");
    float smooth_threshold = 0.5;
    po.Register("smooth-threshold", &smooth_threshold, "smooth threshold for speech interval");
    float min_duration = 2.0;
    po.Register("min-duration", &min_duration, "min duration for speech interval");
    std::string wav_out = "wav_out.wav";
    po.Register("wav-out", &wav_out, "output wav file");
    po.Read(argc, argv);

    std::cout << "* frame len            : " << frame_len << std::endl;
    std::cout << "* frame shift          : " << frame_shift << std::endl;
    std::cout << "* energy thresh        : " << energy_thresh << std::endl;
    std::cout << "* sil to speech_trigger: " << sil_to_speech_trigger << std::endl;
    std::cout << "* speech to sil trigger: " << speech_to_sil_trigger << std::endl;
    std::cout << "* wav bin              : " << wav_bin << std::endl;
    std::cout << "* text out             : " << text_out << std::endl;
    std::cout << "* smooth threshold     : " << smooth_threshold << std::endl;
    std::cout << "* min duration         : " << min_duration << std::endl;

    WaveReader reader(wav_bin.c_str(), 1, 16);
    int sample_rate = reader.SampleRate();
    #ifdef DEBUG
    std::cout << "sample rate: " << sample_rate << std::endl;
    #endif
    
    int num_sample = reader.NumSample();
    int num_point_per_frame = (int)(frame_len * sample_rate);
    int num_point_shift = (int)(frame_shift * sample_rate);

    float *data = (float *)calloc(sizeof(float), num_sample);
    for (int i = 0; i < num_sample; i++) {
        data[i] = reader.Data()[i * reader.NumChannel()];
    }

    Vad vad(energy_thresh, sil_to_speech_trigger, speech_to_sil_trigger);

    int num_frames = (num_sample - num_point_per_frame) / num_point_shift + 1;
    std::vector<int> vad_result;
    std::vector<std::pair<int, int>> speech_intervals; // 存储说话片段的区间，格式为 (起始帧, 终止帧)
    int num_speech_frames = 0;

    for (int i = 0; i < num_sample; i += num_point_shift) {
        if (i + num_point_per_frame > num_sample) break;
        int tags = vad.IsSpeech(data+i, num_point_per_frame) ? 1 : 0;
        vad_result.push_back(tags);
        if (tags == 1) num_speech_frames++;
    }

    // 平滑逻辑：合并近的区间
    for (int i = 0; i < vad_result.size(); i++) {
        if (vad_result[i] == 1) {
            int start_frame = i;
            while (i < vad_result.size() && vad_result[i] == 1) {
                i++;
            }
            int end_frame = i - 1;
            speech_intervals.push_back(std::make_pair(start_frame, end_frame));
        }
    }

    // 平滑逻辑：合并距离小于指定阈值的区间
    for (int i = 0; i < speech_intervals.size() - 1; i++) {
        int cur_end_frame = speech_intervals[i].second;
        int next_start_frame = speech_intervals[i + 1].first;
        if ((next_start_frame - cur_end_frame)*frame_shift <= smooth_threshold) {
            speech_intervals[i].second = speech_intervals[i + 1].second;
            speech_intervals.erase(speech_intervals.begin() + i + 1);
            for (int j = cur_end_frame + 1; j <= next_start_frame; j++) {
                vad_result[j] = 1;
            }
            i--;
            #ifdef DEBUG
            std::cout << "Merge speech interval: " << cur_end_frame << " to " << next_start_frame << std::endl;
            std::cout << "Duration: " << (next_start_frame - cur_end_frame) * frame_shift << std::endl;
            #endif
        }
    }

    // std::string base_name = wav_out.substr(0, wav_out.rfind(".wav"));
    int file_index = 0;

    // 二维数组，存储每个片段的起止时间
    std::vector<std::vector<float>> speech_time;

    std::ofstream fout(text_out, std::ios::app);
    // 平滑逻辑：将长度小于指定时长的片段修改为非人声
    for (int i = 0; i < speech_intervals.size(); i++) {
        int start_frame = speech_intervals[i].first;
        int end_frame = speech_intervals[i].second;
        if ((end_frame - start_frame + 1) * frame_shift < min_duration ) {
        } else {
        #ifdef DEBUG
        std::cout << "# Not remove: " << start_frame << " to " << end_frame << std::endl;
        std::cout << "\tFrom " << start_frame* frame_shift << " to " << end_frame * frame_shift + frame_len << std::endl;
        std::cout << "\tDuration: " << (end_frame - start_frame + 1) * frame_shift << std::endl;
        #endif
        // 计算时间戳
        float start_time = start_frame * frame_shift;
        float end_time = end_frame * frame_shift + frame_len;

        // write to text_out
        
        fout << start_time << "," << end_time << std::endl;
        std::vector <float> temp;
        temp.push_back(start_time);
        temp.push_back(end_time);
        speech_time.push_back(temp);
        }
    }
    fout.close();

    //  待补充
    // 将所有start_time到end_time的片段合成一个音频写出到wav_out
    // 遍历 speech_time

    // 初始化一个向量来存储合并后的音频数据
    std::vector<float> merged_audio_data;

    // 遍历speech_time中的每个时间段
    for (const auto& time_pair : speech_time) {
        float start_time = time_pair[0];
        float end_time = time_pair[1];

        // 计算开始和结束样本的索引
        int start_sample = static_cast<int>(start_time * sample_rate);
        int end_sample = static_cast<int>(end_time * sample_rate);

        // 确保索引不超出范围
        start_sample = std::max(start_sample, 0);
        end_sample = std::min(end_sample, num_sample);

        // 将时间段对应的样本添加到merged_audio_data
        for (int i = start_sample; i < end_sample; ++i) {
            merged_audio_data.push_back(data[i]);
        }
    }

    // 将std::vector转换为动态分配的数组
    float* speech_data = new float[merged_audio_data.size()];
    std::copy(merged_audio_data.begin(), merged_audio_data.end(), speech_data);

    // 计算合并后音频数据的样本数量
    int num_speech_sample = merged_audio_data.size();

    // 使用WaveWriter将合并后的音频数据写入到WAV文件中
    WaveWriter writer(speech_data, num_speech_sample, 1, 16);  // 假设音频为单通道，16位深
    writer.Write(wav_out.c_str());

    // 释放资源
    free(data);
    delete[] speech_data;

    return 0;
}
