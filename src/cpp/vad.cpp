#include <vector>
#include <cmath>
#include <fstream>

using DType = float;    // 定义数据类型，可根据实际情况更改

extern "C" DType vad(const DType* input, size_t input_len, DType energy_threshold, 
                     int mean_window_size, const char* output_file_path) {
    const int frame_size = 256;         // 帧长
    const int frame_shift = 128;        // 帧移
    const int sample_rate = 8000;       // 采样率
    const int fft_size = 512;           // FFT 大小
    const int energy_index = 3;         // 能量谱的第 3 个元素为能量值

    std::vector<DType> output;          // 存储 vad 后的音频数据
    output.reserve(input_len);          // 预先分配输出空间，避免重复调整大小

    std::ofstream ofs(output_file_path, std::ios::out | std::ios::binary);   // 指定输出文件路径和二进制写入方式

    int frame_num = 0, frame_len = input_len;
    while (frame_len >= frame_size) {
        std::vector<DType> frame(input + frame_num * frame_shift, 
                                 input + frame_num * frame_shift + frame_size);

        // 计算帧的能量谱并进行平滑处理
        std::vector<DType> energy(fft_size / 2 + 1, 0);
        for (int i = 0; i < frame_size; ++i) {
            energy[i] = frame[i] * frame[i];
        }
        for (int i = 1; i <= fft_size / 2; ++i) {
            energy[i] = std::sqrt(energy[i]);
        }
        std::vector<DType> smoothed_energy(fft_size / 2 + 1 - mean_window_size, 0);
        for (int i = mean_window_size; i <= fft_size / 2; ++i) {
            DType sum = 0;
            for (int j = i - mean_window_size + 1; j <= i; ++j) {
                sum += energy[j];
            }
            smoothed_energy[i - mean_window_size] = sum / mean_window_size;
        }

        // 判断该帧是否为有声音信号
        if (smoothed_energy[energy_index] > energy_threshold) {
            for (int i = 0; i < frame_size; ++i) {
                output.push_back(frame[i]);
            }
        }

        frame_len -= frame_shift;
        ++frame_num;
    }

    // 写入二进制数据到指定的输出文件中
    ofs.write(reinterpret_cast<const char*>(&output[0]), output.size() * sizeof(DType));

    // 返回 vad 后的音频时长
    return static_cast<DType>(output.size()) / sample_rate;
}