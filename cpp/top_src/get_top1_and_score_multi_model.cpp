// # coding = utf-8
// # @Time    : 2023-11-01  09:06:00
// # @Author  : zhaosheng@nuaa.edu.cn
// # @Describe: Get Model Result from fusion all models.

#include <iostream>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>
#include <malloc.h>
#include "timer.h"
#include "search_best.h"
#include <unordered_map>

#define ALGIN                (32) // 使用SIMD需要内存对齐，128bit的指令需要16位对齐，256bit的指令需要32位对齐

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

typedef float DType;

using namespace std;

void print_help()
{
    cout << "Usage:" << endl;
    cout << "\tARG1: FEATSIZE_list       :   [list] size of each vector, separated by comma" << endl;
    cout << "\tARG2: DB1_list            :   [list] path to Black vector database files for DB1(Test), separated by comma" << endl;
    cout << "\tARG3: DB2_list            :   [list] path to Black vector database files for DB2(Enroll), separated by comma" << endl;
    cout << "\tARG4: ID1                 :   [str] path to the list of ID for Black vectors file for DB1(Test)" << endl;
    cout << "\tARG5: ID2                 :   [str] path to the list of ID for Black vectors file for DB2(Enroll)" << endl;
    cout << "\tARG6: OUTPUT_PATH         :   [str] path to output file" << endl;
    cout << "\t\t\tNote: *1: waiting for test; *2: enroll data" << endl;
}


vector<string> split_string(const string &str, char delimiter)
{
    // cout << "Start spilt " << str << endl;
    vector<string> result;
    stringstream ss(str);
    string token;
    while (getline(ss, token, delimiter))
    {
        result.push_back(token);
    }
    // cout << "End spilt " << str << endl;
    return result;
}

int get_file_lines(const string &file_path)
{
    ifstream file(file_path);
    int lines = 0;
    string line;
    while (getline(file, line))
    {
        lines++;
    }
    return lines;
}

int check_all_values_are_the_same(const int* arr, int size)
{
    int first_value = arr[0];
    for (int i = 1; i < size; i++)
    {
        if (arr[i] != first_value)
        {
            return -1;
        }
    }
    return 0;
}

float get_float_list_mean(const float* arr, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += arr[i];
    }
    // cout << "Get sum: " << sum << " From Arr: " << arr[0] << " , "  << arr[1] << " , " << arr[2] << endl;
    return sum / size;
}

//format print argc and argv
void print_argc_argv(int argc, char* argv[])
{
    cout << "argc: " << argc << endl;
    for (int i = 0; i < argc; i++)
    {
        cout << "argv[" << i << "]: " << argv[i] << endl;
    }
}


int main(int argc, char* argv[])
{
    if (argc < 2 || strcmp(argv[1], "--h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_help();
        return 0;
    }
    if (argc < 7) {
        std::cout << "Error: Not enough arguments provided. Use the \"--help\" option to see usage instructions." << endl;
        return -1;
    }
    if (argc > 7) {
        std::cout << "Error: Too many arguments provided. Use the \"--help\" option to see usage instructions." << endl;
        return -1;
    }

    print_argc_argv(argc, argv);

    const char* FEATSIZE_list = argv[1];
    const char* vector_1_list = argv[2];
    const char* vector_2_list = argv[3];
    const char* id_1_file = argv[4];
    const char* id_2_file = argv[5];
    const char* output_path = argv[6];

    vector<string> FEATSIZE_list_str = split_string(FEATSIZE_list, ',');
    cout << "FEATSIZE list: " << endl;
    for (int i = 0; i < FEATSIZE_list_str.size(); i++)
    {
        cout << "    " << FEATSIZE_list_str[i] << endl;
    }
    vector<string> vector_1_files = split_string(vector_1_list, ',');
    cout << "vector_1_files list: " << endl;
    for (int i = 0; i < vector_1_files.size(); i++)
    {
        cout << "    " << vector_1_files[i] << endl;
    }
    vector<string> vector_2_files = split_string(vector_2_list, ',');
    cout << "vector_2_files list: " << endl;
    for (int i = 0; i < vector_2_files.size(); i++)
    {
        cout << "    " << vector_2_files[i] << endl;
    }

    if ( vector_2_files.size() != vector_1_files.size())
    {
        cout << "Error: The number of Vector 1 database files and Vector 2 database should be the same." << endl;
        return -1;
    }
    

    // For each file vector_2_files
    int VOICENUM_2 = get_file_lines(id_2_file);
    cout << "* Register VOICENUM (VOICENUM2): " << VOICENUM_2 << endl;
    DType* pDB_2_list[vector_2_files.size()];
    std::string id_list_2[VOICENUM_2];
    int FEATSIZE2;
    int FEATSIZE1;
    for (int vector_2_file_index = 0; vector_2_file_index < vector_2_files.size(); vector_2_file_index++)
    {
        FEATSIZE2 = stoi(FEATSIZE_list_str[vector_2_file_index]);
        DType* pDB_2 = reinterpret_cast<DType*>(memalign(ALGIN, sizeof(DType)*VOICENUM_2*FEATSIZE2));
        if(!pDB_2) {
            std::cout << "out of memory\n";
            return -1;
        }
        FILE* fp_vector_2 = fopen(vector_2_files[vector_2_file_index].c_str(), "rb");
        if(!fp_vector_2) {
            std::cout << "open Register(2) vector file: "<<vector_2_files[vector_2_file_index]<<" failed.\n";
            return -1;
        }
        fread(pDB_2, sizeof(DType), VOICENUM_2*FEATSIZE2, fp_vector_2);
        fclose(fp_vector_2);
        FILE* fp_id_file_2 = fopen(id_2_file, "r");
        if(!fp_id_file_2) {
            std::cout << "open file Register(2) id file: "<<id_2_file<<" failed.\n";
            return -1;
        }
        for (int voicenum_2_index = 0; voicenum_2_index < VOICENUM_2; voicenum_2_index++) {
            char id[256];
            fscanf(fp_id_file_2, "%s", id);
            id_list_2[voicenum_2_index] = id;
        }
        pDB_2_list[vector_2_file_index] = pDB_2;
    }
    std::cout << "Read All Register(2) Files !!" << endl;


    // Write result to file
    FILE* fp_output = fopen(output_path, "w");
    if(!fp_output) {
        std::cout << "open OUTPUT FILE: "<<output_path<<" failed.\n";
        return -1;
    }
    std::unordered_map<int, std::pair<std::string, float>> top_similarities;
    
    
    // For each file vector_1_files
    // Get VOICENUM_1_list and pDB_1_list and id_list_1_list
    // ========================================================================================

    int VOICENUM_1 = get_file_lines(id_1_file);
    cout << "* Test VOICENUM (VOICENUM1): " << VOICENUM_1 << endl;
    DType* pDB_1_list[vector_1_files.size()];
    std::string id_list_1[VOICENUM_1];
    for (int vector_1_file_index = 0; vector_1_file_index < vector_1_files.size(); vector_1_file_index++)
    {
        FEATSIZE1 = stoi(FEATSIZE_list_str[vector_1_file_index]);
        DType* pDB_1 = reinterpret_cast<DType*>(memalign(ALGIN, sizeof(DType)*VOICENUM_1*FEATSIZE1));
        if(!pDB_1) {
            std::cout << "out of memory\n";
            return -1;
        }
        FILE* fp_vector_1 = fopen(vector_1_files[vector_1_file_index].c_str(), "rb");
        if(!fp_vector_1) {
            std::cout << "open Test(1) vector file: "<<vector_1_files[vector_1_file_index]<<" failed.\n";
            return -1;
        }

        fread(pDB_1, sizeof(DType), VOICENUM_1*FEATSIZE1, fp_vector_1);
        fclose(fp_vector_1);
        FILE* fp_id_file_1 = fopen(id_1_file, "r");
        if(!fp_id_file_1) {
            std::cout << "open Test(1) id file: "<<id_1_file<<"  failed.\n";
            return -1;
        }
        for (int voicenum_1_index = 0; voicenum_1_index < VOICENUM_1; voicenum_1_index++) {
            char id[256];
            fscanf(fp_id_file_1, "%s", id);
            id_list_1[voicenum_1_index] = id;
        }

        // ========================================================================================
        pDB_1_list[vector_1_file_index] = pDB_1;
    }

    std::cout << "Read All Test(1) Files !!" << std::endl;

    for(int test_index = 0; test_index < VOICENUM_1; test_index++) {
        float max_similarity = 0;
        std::string max_similarity_id = "";
        for(int register_index = 0; register_index < VOICENUM_2; register_index++) {
            int FEATSIZE_auto;
            float similarity_list[vector_1_files.size()];
            for (int pdb_index = 0; pdb_index < vector_1_files.size(); pdb_index++){
                FEATSIZE_auto = stoi(FEATSIZE_list_str[pdb_index]);
                float similarity = Cosine_similarity(pDB_1_list[pdb_index] + test_index*FEATSIZE_auto, pDB_2_list[pdb_index] + register_index*FEATSIZE_auto, FEATSIZE_auto);
                similarity_list[pdb_index] = similarity;
            }
            float similarity = get_float_list_mean(similarity_list, vector_1_files.size());
            
            if (similarity > max_similarity) {
                max_similarity = similarity;
                max_similarity_id = id_list_2[register_index];
            }
            top_similarities[test_index] = std::make_pair(max_similarity_id, max_similarity);
        }
    }

    // write top similarities to file
    for (int test_index = 0; test_index < VOICENUM_1; test_index++) {
        std::string str = top_similarities[test_index].first + "," + id_list_1[test_index] + "," + std::to_string(top_similarities[test_index].second) + "\n";
        fwrite(str.c_str(), sizeof(char), str.size(), fp_output);
    }

    // Release memory
    for (int i = 0; i < vector_1_files.size(); i++)
    {
        free(pDB_1_list[i]);
        free(pDB_2_list[i]);
    }
    return 0;
}