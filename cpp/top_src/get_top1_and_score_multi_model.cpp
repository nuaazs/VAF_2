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
    cout << "Usage: program_name NUM_BLACK EMB_SIZE DB1 ID1 DB2 ID2 OUTPUT_PATH" << endl;
    cout << "Options:" << endl;
    cout << "\tEMB_list      :   size of each vector, separated by comma" << endl;
    cout << "\tDB1_list      :   path to Black vector database files for DB1, separated by comma" << endl;
    cout << "\tDB2_list      :   path to Black vector database files for DB2, separated by comma" << endl;
    cout << "\tID1           :   path to the list of IDs for Black vectors file for DB1" << endl;
    cout << "\tID2           :   path to the list of IDs for Black vectors file for DB2" << endl;
    cout << "\tOUTPUT_PATH   :   path to output file" << endl;
    cout << "\t*1: waiting for test; *2: enroll data" << endl;
}


vector<string> split_string(const string &str, char delimiter)
{
    cout << "Start spilt " << str << endl;
    vector<string> result;
    stringstream ss(str);
    string token;
    while (getline(ss, token, delimiter))
    {
        result.push_back(token);
    }
    cout << "End spilt " << str << endl;
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
    cout<<"FEATSIZE: "<<FEATSIZE_list<<endl;
    cout<<"vector_1_list: "<<vector_1_list<<endl;
    cout<<"vector_2_list: "<<vector_2_list<<endl;
    cout<<"id_1_file: "<<id_1_file<<endl;
    cout<<"id_2_file: "<<id_2_file<<endl;
    cout<<"output_path: "<<output_path<<endl;

    vector<string> FEATSIZE_list_str = split_string(FEATSIZE_list, ',');
    cout << "FEATSIZE_list_str: " << FEATSIZE_list_str[0] << endl;
    // change FEATSIZE_list_str to int
    
    vector<string> vector_1_files = split_string(vector_1_list, ',');
    vector<string> vector_2_files = split_string(vector_2_list, ',');
    // vector<string> id_1_files = split_string(id_1_list, ',');
    // vector<string> id_2_files = split_string(id_2_list, ',');

    if ( vector_2_files.size() != vector_1_files.size())
    {
        cout << "Error: The number of Vector 1 database files and Vector 2 database should be the same." << endl;
        return -1;
    }
    

    // For each file vector_2_files

    int VOICENUM_2 = get_file_lines(id_2_file);
    DType* pDB_2_list[vector_2_files.size()];
    std::string id_list_2[VOICENUM_2];
    int FEATSIZE2;
    int FEATSIZE1;
    for (int j = 0; j < vector_2_files.size(); j++)
    {
        cout << "* VOICENUM_2: " << VOICENUM_2 << endl;
        FEATSIZE2 = stoi(FEATSIZE_list_str[j]);
        cout<<  "* FEATSIZE_2: " << FEATSIZE2  <<endl;
        
        // Read DB 2
        // ========================================================================================
        DType* pDB_2 = reinterpret_cast<DType*>(memalign(ALGIN, sizeof(DType)*VOICENUM_2*FEATSIZE2));
        if(!pDB_2) {
            std::cout << "out of memory\n";
            return -1;
        }

        // read bin file: vector_2_files[j]
        FILE* fp_vector_2 = fopen(vector_2_files[j].c_str(), "rb");
        if(!fp_vector_2) {
            std::cout << "open file a.bin: "<<vector_2_files[j]<<" cjsd failed.\n";
            return -1;
        }

        fread(pDB_2, sizeof(DType), VOICENUM_2*FEATSIZE2, fp_vector_2);
        fclose(fp_vector_2);
        
        FILE* fp_id_file_2 = fopen(id_2_file, "r");
        if(!fp_id_file_2) {
            std::cout << "open file a.bin: "<<id_2_file<<" cjsd failed.\n";
            return -1;
        }
        for (int i = 0; i < VOICENUM_2; i++) {
            char id[256];
            fscanf(fp_id_file_2, "%s", id);
            id_list_2[i] = id;
        }
        // append to pDB_2_list
        pDB_2_list[j] = pDB_2;
        // ========================================================================================
    }

    std::cout << "Read FEATSIZE 2 Done." << endl;


    // Write result to file
    // ========================================================================================
    FILE* fp_output = fopen(output_path, "w");
    if(!fp_output) {
        std::cout << "open file a.bin: "<<output_path<<" failed.\n";
        return -1;
    }
    std::unordered_map<int, std::pair<std::string, float>> top_similarities;
    
    

    // For each file vector_1_files
    // Get VOICENUM_1_list and pDB_1_list and id_list_1_list
    // ========================================================================================

    int VOICENUM_1 = get_file_lines(id_1_file);
    DType* pDB_1_list[vector_1_files.size()];
    std::string id_list_1[VOICENUM_1];

    for (int i =0; i < vector_1_files.size(); i++)
    {
        FEATSIZE1 = stoi(FEATSIZE_list_str[i]);
        cout<<  "* FEATSIZE_1: " << FEATSIZE1  <<endl;
        
        DType* pDB_1 = reinterpret_cast<DType*>(memalign(ALGIN, sizeof(DType)*VOICENUM_1*FEATSIZE1));

        // Read DB 1
        // ========================================================================================
        if(!pDB_1) {
            std::cout << "out of memory\n";
            return -1;
        }
        // read bin file: vector_1_files[i]
        FILE* fp_vector_1 = fopen(vector_1_files[i].c_str(), "rb");
        if(!fp_vector_1) {
            std::cout << "open file a.bin: "<<vector_1_files[i]<<" cjsd failed.\n";
            return -1;
        }

        for (int i = 0; i < VOICENUM_1; i++) {
            std::vector<DType> feature_i(FEATSIZE1);
            if (fread(&feature_i[0], sizeof(DType), FEATSIZE1, fp_vector_1) != FEATSIZE1) {
                std::cout << "read feature failed.\n";
                // cout shape diff
                cout << "i: " << i << endl;
                cout << "feature_i.size(): " << feature_i.size() << endl;
                cout << "FEATSIZE1: " << FEATSIZE1 << endl;
                return -1;
            }
            for (int j = 0; j < FEATSIZE1; j++) {
                pDB_1[i * FEATSIZE1 + j] = feature_i[j];
            }
        }
        fclose(fp_vector_1);
        
        
        FILE* fp_id_file_1 = fopen(id_1_file, "r");
        if(!fp_id_file_1) {
            std::cout << "open file a.bin: "<<id_1_file<<"  failed.\n";
            return -1;
        }
        for (int i = 0; i < VOICENUM_1; i++) {
            char id[256];
            fscanf(fp_id_file_1, "%s", id);
            id_list_1[i] = id;
        }

        // ========================================================================================
        pDB_1_list[i] = pDB_1;
    }

    std::cout << "All Embedding Read Okay!" << std::endl;
    for(int j = 0; j < VOICENUM_1; j++) {
        float max_similarity = 0;
        std::string max_similarity_id = "";
        for(int i = 0; i < VOICENUM_2; i++) {
            // j: index of vector_1_files, i: index of vector_2_files
            // j: ID of Test, i: ID of Enroll

            // calc cosine similarity
            // init similarity list to calc mean similarity
            int FEATSIZE_auto;
            float similarity_list[vector_1_files.size()];
            for (int pdb_index = 0; pdb_index < vector_1_files.size(); pdb_index++){
                FEATSIZE_auto = stoi(FEATSIZE_list_str[pdb_index]);
                float similarity = Cosine_similarity(pDB_1_list[pdb_index] + i*FEATSIZE_auto, pDB_2_list[pdb_index] + j*FEATSIZE_auto, FEATSIZE_auto);
                similarity_list[pdb_index] = similarity;
            }
            float similarity = get_float_list_mean(similarity_list, vector_1_files.size());
            
            if (similarity > max_similarity) {
                max_similarity = similarity;
                max_similarity_id = id_list_1[i];
            }
            top_similarities[j] = std::make_pair(max_similarity_id, max_similarity);
        }
    }

    // write top similarities to file
    for (int j = 0; j < VOICENUM_1; j++) {
        std::string str = top_similarities[j].first + "," + id_list_2[j] + "," + std::to_string(top_similarities[j].second) + "\n";
        fwrite(str.c_str(), sizeof(char), str.size(), fp_output);
    }
    // Release memory
    for (int i = 0; i < vector_1_files.size(); i++)
    {
        free(pDB_1_list[i]);
    }
    for (int i = 0; i < vector_2_files.size(); i++)
    {
        free(pDB_2_list[i]);
    }

    return 0;
}
