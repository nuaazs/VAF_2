/* Created on 2016-09-14
 * Author: Binbin Zhang
 * About: cpp command argv parser, 
 */

#ifndef PARSE_OPTION_H_
#define PARSE_OPTION_H_

#include <stdlib.h>
#include <assert.h>

#include <string>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>


class ParseOptions {
public:
    ParseOptions(const char *usage): usage_(usage) {} 

    ~ParseOptions() {};

    int NumArgs() {
        return args_.size();
    }

    std::string GetArg(int i) {
        if (i < 1 || i > args_.size()) {
            std::cerr << "Invalid arg index " << i << "\n";
            exit(1);
        }
        return args_[i - 1];
    }

    int Read(int argc, const char *const *argv) {
        argc_ = argc;
        argv_ = argv;
        std::string key, value;
        int i = 0;
        bool has_equal_sign;
        // argv_[0], program name
        for (i = 1; i < argc; i++) {
            if (std::strncmp(argv[i], "--", 2) == 0) {
                bool has_equal_sign;
                SplitLongArg(argv[i], &key, &value, &has_equal_sign);
                Trim(&value);
                if (key.compare("help") == 0) {
                    PrintUsage();
                    exit(0);
                }
                SetOption(key, value, has_equal_sign);
            }
            else break;
        }
        // the left are standard argument
        for (; i < argc; i++) {
            args_.push_back(argv[i]);
        }
    }

    void PrintUsage() {
        std::cerr << "\n" << usage_ << "\n";
        std::cerr << "Options:\n";
        std::map<std::string, std::string>::iterator it = doc_map_.begin();
        for (; it != doc_map_.end(); it++) {
            std::cerr << "  --" << std::setw(25) << std::left << it->first
                      << " : " << it->second << '\n';
        }
    }

    bool SetOption(const std::string &key,
            const std::string &value,
            bool has_equal_sign) {
        if (bool_map_.end() != bool_map_.find(key)) {
            if (has_equal_sign && value == "") {
                std::cerr << "Invalid option --" << key << " =\n";
                PrintUsage();
                exit(1);
            }
            *(bool_map_[key]) = ToBool(value);
            return true;
        }
        // otherwise has no value
        if (!has_equal_sign) {
            std::cerr << "Invalid option -- " << key << "\n";
            PrintUsage();
            exit(1);
        }

        if (int_map_.end() != int_map_.find(key)) {
            // atoi is not reliable
            *(int_map_[key]) = atoi(value.c_str());
        } else if (float_map_.end() != float_map_.find(key)) {
            // atof is not reliable
            *(float_map_[key]) = (float)atof(value.c_str());
        } else if (string_map_.end() != string_map_.find(key)) {
            *(string_map_[key]) = value;
        } else {
            std::cerr << "Invalid option -- " << key << "\n";
            PrintUsage();
            exit(1);
        }
        return true;
    }

    bool ToBool(std::string str) {
        // allow "" as a valid option for "true", so that --x is the same as --x=true
        if ((str.compare("true") == 0) || (str.compare("t") == 0)
                || (str.compare("1") == 0) || (str.compare("") == 0)) {
            return true;
        }
        if ((str.compare("false") == 0) || (str.compare("f") == 0)
                || (str.compare("0") == 0)) {
            return false;
        }
        std::cerr << "Invalid format for boolean argument [expected true or false]: "
                  << str << "\n";
        return false;  // otherwise
    }

    // TODO, safe convert to int and float
    int ToInt(std::string str) {
    }
    int ToFloat(std::string str) {
    }

    void SplitLongArg(std::string in, std::string *key,
            std::string *value, bool *has_equal_sign) {
        assert(in.substr(0, 2) == "--");  // precondition.
        size_t pos = in.find_first_of('=', 0);
        if (pos == std::string::npos) {  // we allow --option for bools
            // defaults to empty.  We handle this differently in different cases.
            *key = in.substr(2, in.size()-2);  // 2 because starts with --.
            *value = "";
            *has_equal_sign = false;
        } else if (pos == 2) {  // we also don't allow empty keys: --=value
            PrintUsage();
            std::cerr << "Invalid option (no key): " << in;
            exit(1);
        } else {  // normal case: --option=value
            *key = in.substr(2, pos-2);  // 2 because starts with --.
            *value = in.substr(pos + 1);
            *has_equal_sign = true;
        }
    }

    void Trim(std::string *str) {
        const char *white_chars = " \t\n\r\f\v";

        std::string::size_type pos = str->find_last_not_of(white_chars);
        if (pos != std::string::npos)  {
            str->erase(pos + 1);
            pos = str->find_first_not_of(white_chars);
            if (pos != std::string::npos) str->erase(0, pos);
        } else {
            str->erase(str->begin(), str->end());
        }     
    } 

    // Methods from the interface
    void Register(const std::string &name,
            bool *ptr, const std::string &doc) {
        assert(ptr != NULL);
        bool_map_[name] = ptr;
        std::ostringstream ss;
        std::string b = (*ptr) ? "true" : "false";
        ss << doc << " (bool, default = " << b << ")";
        doc_map_[name] = ss.str();
    }

    void Register(const std::string &name,
            int *ptr, const std::string &doc) {
        assert(ptr != NULL);
        int_map_[name] = ptr;
        std::ostringstream ss;
        ss << doc << " (int, default = " << *ptr << ")";
        doc_map_[name] = ss.str();
    }

    void Register(const std::string &name,
            float *ptr, const std::string &doc) {
        assert(ptr != NULL);
        float_map_[name] = ptr;
        std::ostringstream ss;
        ss << doc << " (float, default = " << *ptr << ")";
        doc_map_[name] = ss.str();
    }

    void Register(const std::string &name,
            std::string *ptr, const std::string &doc) {
        assert(ptr != NULL);
        string_map_[name] = ptr;
        std::ostringstream ss;
        ss << doc << " (string, default = " << *ptr << ")";
        doc_map_[name] = ss.str();
    }

private:
    // maps for option variables
    std::map<std::string, bool*> bool_map_;
    std::map<std::string, int*> int_map_;
    std::map<std::string, float*> float_map_;                                                   
    std::map<std::string, std::string*> string_map_;
    // document map
    std::map<std::string, std::string> doc_map_;
    const char *usage_;
    int argc_;                                                                                  
    const char *const *argv_; 
    std::vector<const char *> args_;
};


#endif
