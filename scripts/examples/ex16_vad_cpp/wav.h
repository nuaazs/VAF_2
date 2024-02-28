/* Created on 2016-08-15
 * Author: Binbin Zhang
 * Modified on 2023-06-17 by Sheng Zhao
 */

#ifndef WAVE_H_
#define WAVE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>

class WaveReader {
public:
    WaveReader(const char *filename, const int num_channel, const int bit_depth) {
        FILE *fp = fopen(filename, "rb");
        if (NULL == fp) {
            perror(filename);
            exit(1);
        }
        
        fseek(fp, 0, SEEK_END);
        int file_size = ftell(fp);
        int num_data = file_size / (bit_depth / 8);
        data_ = new float[num_data];
        num_sample_ = num_data / num_channel;
       
        fseek(fp, 0, SEEK_SET);
        for (int i = 0; i < num_data; i++) {
            switch (bit_depth) {
                case 8: {
                    char sample;
                    fread(&sample, 1, sizeof(char), fp);
                    data_[i] = (float)sample;
                    break;
                }
                case 16: {
                    short sample;
                    fread(&sample, 1, sizeof(short), fp);
                    data_[i] = (float)sample;
                    break;
                }
                case 32: {
                    int sample;
                    fread(&sample, 1, sizeof(int), fp);
                    data_[i] = (float)sample;
                    break;
                }
                default:
                    fprintf(stderr, "unsupported quantization bits");
                    exit(1);
            }
        }
        fclose(fp);
        num_channel_ = num_channel;
        bit_depth_ = bit_depth;
    }

    int NumChannel() const { return num_channel_; }
    int BitDepth() const { return bit_depth_; }
    int NumSample() const { return num_sample_; }
    int SampleRate() const { return (int)8000; }
    int BitsPerSample() const { return (int)16; }

    ~WaveReader() {
        if (data_ != NULL) delete[] data_;
    }

    const float *Data() const { return data_; }

private:
    int num_channel_;
    int bit_depth_;
    int num_sample_; // sample points per channel
    float *data_;
};

class WaveWriter {
public:
    WaveWriter(const float *data, int num_sample, 
              int num_channel, int bit_depth):
        data_(data), num_sample_(num_sample), 
        num_channel_(num_channel),
        bit_depth_(bit_depth) {}

    void Write(const char *filename) {
        FILE *fp = fopen(filename, "wb");
        if (NULL == fp) {
            perror("open file failed");
            exit(1);
        }
        
        int chunk_size = num_sample_ * num_channel_ * (bit_depth_ / 8);
        int wave_size = 4 + 8 + 16 + 8 + chunk_size;
        fwrite("RIFF", 1, 4, fp); // RIFF header
        fwrite(&wave_size, 1, sizeof(int), fp); // wave_size
        fwrite("WAVEfmt ", 2, 4, fp); // WAVE header
        int format_size = 16;
        fwrite(&format_size, 1, sizeof(int), fp); // format size
        short format_tag = 1; // PCM
        fwrite(&format_tag, 1, sizeof(short), fp); // format tag
        short channels = num_channel_;
        fwrite(&channels, 1, sizeof(short), fp); // channels
        int samples_per_sec = 8000;
        fwrite(&samples_per_sec, 1, sizeof(int), fp); // samples per sec
        int bytes_per_sec = samples_per_sec * num_channel_ * (bit_depth_ / 8);
        fwrite(&bytes_per_sec, 1, sizeof(int), fp); // bytes per sec
        short block_align = num_channel_ * (bit_depth_ / 8);
        fwrite(&block_align, 1, sizeof(short), fp); // block align
        fwrite(&bit_depth_, 1, sizeof(short), fp); // bits per sample
        fwrite("data", 1, 4, fp); // data header
        fwrite(&chunk_size, 1, sizeof(int), fp); // chunk size
        
        for (int i = 0; i < num_sample_; i++) {
            for (int j = 0; j < num_channel_; j++) {
                switch (bit_depth_) {
                    case 8: {
                        char sample = (char)data_[i * num_channel_ + j];
                        fwrite(&sample, 1, sizeof(sample), fp);
                        break;
                    }
                    case 16: {
                        short sample = (short)data_[i * num_channel_ + j];
                        fwrite(&sample, 1, sizeof(sample), fp);
                        break;
                    }
                    case 32: {
                        int sample = (int)data_[i * num_channel_ + j];
                        fwrite(&sample, 1, sizeof(sample), fp);
                        break;
                    }
                }
            }
        }
        fclose(fp);
    }
private:
    const float *data_;
    int num_sample_; // total float points in data_
    int num_channel_;
    int bit_depth_;
};

#endif