#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

#include <vector>
#include <string>
#include <unordered_map>

namespace wespeaker {

const char WHITESPACE[] = " \n\r\t\f\v";

void WriteToFile(const std::string& file_path,
                 const std::vector<std::vector<float>>& embs);
void ReadToFile(const std::string& file_path,
                std::vector<std::vector<float>>* embs);

// Split the string with space or tab.
void SplitString(const std::string& str, std::vector<std::string>* strs);

void SplitStringToVector(const std::string& full, const char* delim,
                         bool omit_empty_strings,
                         std::vector<std::string>* out);

}  // namespace wespeaker

#endif  // UTILS_UTILS_H_