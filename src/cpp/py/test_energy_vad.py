import numpy as np
import ctypes

import ctypes
import tempfile
lib = ctypes.CDLL('./vad.so')
def vad(input_lst, energy_threshold, mean_window_size, output_file_path):
    # 加载编译生成的 .so 文件


    # 定义函数的参数和返回值类型
    lib.vad.restype = ctypes.c_float
    lib.vad.argtypes = [ctypes.POINTER(ctypes.c_float), 
                        ctypes.c_size_t,
                        ctypes.c_float,
                        ctypes.c_int,
                        ctypes.c_char_p]

    # 将 input_lst 转换为 C++ 可以接受的数据类型
    input_arr = (ctypes.c_float * len(input_lst))(*input_lst)

    # 调用 C++ 函数进行 VAD 算法处理并获取 vad 后的音频时长
    vad_len = lib.vad(input_arr, len(input_lst), energy_threshold, mean_window_size, output_file_path)

    return vad_len
if __name__ == "__main__":
    # read wav data, float32, 8000 sample rate, 1 channel
    wav_file_path = "/ssd2/voiceprint-recognition-system/src/api_test/1p1c8k.wav"
    data = np.fromfile(wav_file_path, dtype=np.float32).reshape(-1).tolist()
    data_npy = np.array(data, dtype=np.float32)
    print(data_npy.shape)
    # result = calculate(input_data, len(data))

    # 调用 VAD 算法
    output = vad(data, 0.5, 512, b"temp.bin")
    print(output)

    # read temp.bin data, float64, 8000 sample rate, 1 channel
    # save to temp.wav
    data = np.fromfile("temp.bin", dtype=np.float64).reshape(-1).tolist()
    data = np.array(data, dtype=np.float32)
    # data.tofile("temp.wav")
    print(data.shape)