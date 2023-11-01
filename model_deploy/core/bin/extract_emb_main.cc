#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdio>

#include "frontend/wav.h"
#include "utils/utils.h"
#include "utils/timer.h"
#include "speaker/speaker_engine.h"

DEFINE_string(wav_list, "", "input wav scp");
DEFINE_string(model_path, "", "model_path");
DEFINE_string(result, "", "output embedding file");

// string FLAGS_model_path="no use";
int FLAGS_fbank_dim=80;
int FLAGS_sample_rate=16000;
int FLAGS_embedding_size=256;
int FLAGS_SamplesPerChunk=160000;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  // mac地址核验
  FILE* fp;
  char buf[100];
  // 调用系统指令获取 MAC 地址
  fp = popen("ifconfig", "r");
  // 读取系统返回的信息，查找 MAC 地址字段
  while (fgets(buf, sizeof(buf), fp)) {
      if (strstr(buf, "ether") != NULL) {
          // 提取 MAC 地址
          char* mac = strtok(buf, " ");
          mac = strtok(NULL, " ");
          // std::cout << "MAC 地址为：" << mac << std::endl;
          // 比较 MAC 地址是否匹配
          if (strcmp(mac, "02:42:e1:fc:9e:e5") == 0) {
              std::cout << "授权信息验证成功！" << std::endl;
          } else {
              std::cout << "授权信息验证失败！" << std::endl;
              return 0;
          }
          break;
      }
  }
  pclose(fp);

  // 获取当前系统时间
  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm* currentTime = std::localtime(&now);
  // 获取当前年、月、日
  int year = currentTime->tm_year + 1900;
  int month = currentTime->tm_mon + 1;
  int day = currentTime->tm_mday;
  // 指定起始和结束日期范围
  int startYear = 2023;
  int startMonth = 10;
  int startDay = 1;
  int endYear = 2023;
  int endMonth = 12;
  int endDay = 30;
  // 检查当前时间是否在指定的日期范围内
  if (year > startYear || (year == startYear && month > startMonth) ||
      (year == startYear && month == startMonth && day >= startDay)) {
      if (year < endYear || (year == endYear && month < endMonth) ||
          (year == endYear && month == endMonth && day <= endDay)) {
          // 在指定日期范围内，执行程序逻辑
          std::cout << "当前时间在有效日期范围内，剩余有效时间: " << endYear - year << "年"
                    << endMonth - month << "月" << endDay - day << "日" << std::endl;
      } else {
          // 不在指定日期范围内
          std::cout << "当前时间不在有效日期范围内。" << std::endl;
          // 退出程序
          return 0;
      }
  } else {
      // 不在指定日期范围内
      std::cout << "当前时间不在有效日期范围内。" << std::endl;
      // 退出程序
      return 0;
  }


  // init model
  LOG(INFO) << "Init model ...";
  auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
    FLAGS_model_path,FLAGS_fbank_dim, FLAGS_sample_rate,
    FLAGS_embedding_size, FLAGS_SamplesPerChunk);
  std::cout << "MODEL LOAD SUCCESS" << std::endl;
  int embedding_size = speaker_engine->EmbeddingSize();
  LOG(INFO) << "embedding size: " << embedding_size;
  // read wav.scp
  // [utt, wav_path]
  std::vector<std::pair<std::string, std::string>> waves;
  std::ifstream wav_scp(FLAGS_wav_list);
  std::string line;
  while (getline(wav_scp, line)) {
    std::vector<std::string> strs;
    wespeaker::SplitString(line, &strs);
    CHECK_EQ(strs.size(), 2);
    waves.emplace_back(make_pair(strs[0], strs[1]));
  }

  std::ofstream result;
  if (!FLAGS_result.empty()) {
    result.open(FLAGS_result, std::ios::out);
  }
  std::ostream &buffer = FLAGS_result.empty() ? std::cout : result;

  int total_waves_dur = 0;
  int total_extract_time = 0;
  for (auto &wav : waves) {
    auto data_reader = wenet::ReadAudioFile(wav.second);
    CHECK_EQ(data_reader->sample_rate(), 16000);
    int16_t* data = const_cast<int16_t*>(data_reader->data());
    int samples = data_reader->num_sample();
    // NOTE(cdliang): memory allocation
    std::vector<float> embs(embedding_size, 0);
    result << wav.first;

    int wave_dur = static_cast<int>(static_cast<float>(samples) /
                                    data_reader->sample_rate() * 1000);
    int extract_time = 0;
    wenet::Timer timer;
    // log data shape
    LOG(INFO) << "data shape: " << data << " " << samples;
    std::cout << "Start Extract ..." << std::endl;
    speaker_engine->ExtractEmbedding(data, samples, &embs);
    std::cout << "End Extract ..." << std::endl;
    extract_time = timer.Elapsed();
    for (size_t i = 0; i < embs.size(); i++) {
      result << " " << embs[i];
    }
    result << std::endl;
    LOG(INFO) << "process: " << wav.first
              << " RTF: " << static_cast<float>(extract_time) / wave_dur;
    total_waves_dur += wave_dur;
    total_extract_time += extract_time;
  }
  result.close();
  LOG(INFO) << "Total: process " << total_waves_dur << "ms audio taken "
            << total_extract_time << "ms.";
  LOG(INFO) << "RTF: "
            << static_cast<float>(total_extract_time) / total_waves_dur;
  return 0;
}