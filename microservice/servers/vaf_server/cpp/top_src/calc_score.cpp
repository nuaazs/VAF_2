#include <iostream>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>
#include <malloc.h>
#include "timer.h"
#include "search_best.h"

#define ALGIN                (32) // 使用SIMD需要内存对齐，128bit的指令需要16位对齐，256bit的指令需要32位对齐

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

typedef float DType;

using namespace std;

void print_help() {
    cout << "Usage: program_name NUM_CJSD NUM_BLACK EMB_SIZE DB1 DB2 ID1 ID2 OUTPUT_PATH" << endl;
    cout << "Options:" << endl;
    cout << "\tNUM_CJSD : number of CJSD vectors" << endl;
    cout << "\tNUM_BLACK : number of Black vectors" << endl;
    cout << "\tEMB_SIZE : size of each vector" << endl;
    cout << "\tDB1 : path to CJSD vector database file" << endl;
    cout << "\tDB2 : path to Black vector database file" << endl;
    cout << "\tID1 : path to the list of IDs for CJSD vectors" << endl;
    cout << "\tID2 : path to the list of IDs for Black vectors" << endl;
    cout << "\tOUTPUT_PATH : path to output file" << endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2 || strcmp(argv[1], "--h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_help();
        return 0;
    }
    if (argc < 6) {
        std::cout << "Error: Not enough arguments provided. Use the \"--help\" option to see usage instructions." << endl;
        return -1;
    }

    const int VOICENUM_CJSD = atoi(argv[1]);
    const int VOICENUM_BLACK = atoi(argv[2]);
    const int FEATSIZE = atoi(argv[3]);
    const char* vectorDB_cjsd_path = argv[4];
    const char* vectorDB_black_path = argv[5];
    const char* cjsd_id_list_path = argv[6];
    const char* black_id_list_path = argv[7];
    const char* output_path = argv[8];

    // CJSD
    DType* pDB_cjsd = reinterpret_cast<DType*>(memalign(ALGIN, sizeof(DType)*VOICENUM_CJSD*FEATSIZE));
    if(!pDB_cjsd) {
        std::cout << "out of memory\n";
        return -1;
    }
    FILE* fp = fopen(vectorDB_cjsd_path, "rb");
    if(!fp) {
        std::cout << "open file vectorDB.bin cjsd failed.\n";
        return -1;
    }
    for (int i = 0; i < VOICENUM_CJSD; i++) {
        std::vector<DType> feature_i(FEATSIZE);
        if (fread(&feature_i[0], sizeof(DType), FEATSIZE, fp) != FEATSIZE) {
            std::cout << "read feature failed.\n";
            return -1;
        }
        for (int j = 0; j < FEATSIZE; j++) {
            pDB_cjsd[i * FEATSIZE + j] = feature_i[j];
        }
    }
    fclose(fp);
    std::string id_list_cjsd[VOICENUM_CJSD];
    FILE* fp1 = fopen(cjsd_id_list_path, "r");
    if(!fp1) {
        std::cout << "open file vector.txt failed.\n";
        return -1;
    }
    for(int i = 0; i < VOICENUM_CJSD; i++) {
        char id[100];
        fscanf(fp1, "%s", id);
        id_list_cjsd[i] = id;
    }


    // BLACK
    DType* pDB_black = reinterpret_cast<DType*>(memalign(ALGIN, sizeof(DType)*VOICENUM_BLACK*FEATSIZE));
    if(!pDB_black) {
        std::cout << "out of memory\n";
        return -1;
    }
    FILE* fp2 = fopen(vectorDB_black_path, "rb");
    if(!fp2) {
        std::cout << "open file vectorDB.bin black failed.\n";
        return -1;
    }
    fread(pDB_black, sizeof(DType), VOICENUM_BLACK*FEATSIZE, fp2);
    fclose(fp2);
    std::string id_list_black[VOICENUM_BLACK];
    FILE* fp4 = fopen(black_id_list_path, "r");
    if(!fp4) {
        std::cout << "open file vector.txt failed.\n";
        return -1;
    }
    for(int i = 0; i < VOICENUM_BLACK; i++) {
        char id[100];
        fscanf(fp4, "%s", id);
        id_list_black[i] = id;
    }

    // Write result to file with progress bar
    const int total = VOICENUM_BLACK * VOICENUM_CJSD;
    float cnt = 0;
    const int barWidth = 70;
    
    // Write result to file
    FILE* fp3 = fopen(output_path, "w");
    if(!fp3) {
        std::cout << "open file result.txt failed.\n";
        return -1;
    }
    for(int i = 0; i < VOICENUM_BLACK; i++) {
        for(int j = 0; j < VOICENUM_CJSD; j++) {
            // calc cosine similarity
            float similarity = Cosine_similarity(pDB_black + i*FEATSIZE, pDB_cjsd + j*FEATSIZE, FEATSIZE);
            // std::cout  << id_list_black[i] << "," << id_list_cjsd[j] << ","<< similarity<< endl;
            // write to file string
            std::string str = id_list_black[i] + "," + id_list_cjsd[j] + "," + std::to_string(similarity) + "\n";
            fwrite(str.c_str(), sizeof(char), str.size(), fp3);
            // print progress bar
            cnt++;
            float progress = cnt / total;
            int pos = barWidth * progress;
            cout << "[";
            for (int k = 0; k < barWidth; ++k) {
                if (k < pos) cout << "=";
                else if (k == pos) cout << ">";
                else cout << " ";
            }
            cout << "] " << int(progress * 100.0) << "%\r";
            fflush(stdout);
        }
    }

    // Release memory
    free(pDB_cjsd);
    free(pDB_black);

    return 0;
}
