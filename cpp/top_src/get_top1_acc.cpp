#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

struct Data {
    std::string id_a;
    std::string id_b;
    double similarity;
};

void output_result(const std::string& path, int TP, int TN, int FP, int FN, double TPR, double FPR, double TNR, double FNR, double ACC, double Precision, double Recall, double F1) {
    std::ofstream res(path);
    res << "| Evaluation Index | Value |" << std::endl;
    res << "| :-- | :--: |" << std::endl;
    res << "| TP | " << TP << " |" << std::endl;
    res << "| TN | " << TN << " |" << std::endl;
    res << "| FP | " << FP << " |" << std::endl;
    res << "| FN | " << FN << " |" << std::endl;
    res << "| TPR | " << TPR << " |" << std::endl;
    res << "| FPR | " << FPR << " |" << std::endl;
    res << "| TNR | " << TNR << " |" << std::endl;
    res << "| FNR | " << FNR << " |" << std::endl;
    res << "| ACC | " << ACC << " |" << std::endl;
    res << "| Precision | " << Precision << " |" << std::endl;
    res << "| Recall | " << Recall << " |" << std::endl;
    res << "| F1 | " << F1 << " |" << std::endl;
    res.close();
}

int main(int argc, char* argv[]) {

    if (argc < 6 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "--h") {
        std::cout << "Usage: " << argv[0] << " txt_file_path th_start th_stop th_step save_dir" << std::endl;
        return 1;
    }

    // 读取txt文件
    std::ifstream fin(argv[1]);
    std::string line;
    std::vector<Data> data_vec;
    while(std::getline(fin, line)) {
        size_t pos1 = 0, pos2 = 0;
        pos1 = line.find(',');
        pos2 = line.find(',', pos1+1);
        std::string id_a = line.substr(0, pos1);
        std::string id_b = line.substr(pos1+1, pos2-pos1-1);
        double similarity = std::stod(line.substr(pos2+1));
        data_vec.push_back({id_a, id_b, similarity});
    }
    fin.close();

    // 获取总数
    int total = static_cast<int>(data_vec.size());

    // 排序获取TOP1
    std::sort(data_vec.begin(), data_vec.end(), [](const Data& d1, const Data& d2){return d1.similarity > d2.similarity;});
    std::string top1 = data_vec[0].id_a.substr(0, data_vec[0].id_a.find('_'));

    // 获取th的起始值、结束值和步长
    double th_start = std::stod(argv[2]);
    double th_stop = std::stod(argv[3]);
    double th_step = std::stod(argv[4]);

    // 创建保存结果的目录
    std::string result_dir = argv[5];
    if (!fs::exists(result_dir)) {
        fs::create_directory(result_dir);
    }

    // 循环计算不同th下的指标并输出结果
    for (double th = th_start; th <= th_stop; th += th_step) {
        //cout now th
        std::cout << "Now th: " << th << std::endl;
        // 计算TP, FP, TN, FN
        int TP = 0, FP = 0, TN = 0, FN = 0;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << th;
        std::string th_str = ss.str();
        std::string result_file_TP = result_dir + "/TP_" + th_str + ".txt";
        std::string result_file_FP = result_dir + "/FP_" + th_str + ".txt";
        std::string result_file_FN = result_dir + "/FN_" + th_str + ".txt";
        std::string result_file_TN = result_dir + "/TN_" + th_str + ".txt";

        std::ofstream fout_TP(result_file_TP);
        std::ofstream fout_FP(result_file_FP);
        std::ofstream fout_FN(result_file_FN);
        std::ofstream fout_TN(result_file_TN);
        for (const auto& data : data_vec) {
            

            if (data.similarity >= th) {
                std::string id_a = data.id_a.substr(0, data.id_a.find('_'));
                std::string id_b = data.id_b.substr(0, data.id_b.find('_'));
                // cout id_a id_b score
                // std::cout << id_a << " " << id_b << " " << data.similarity << std::endl;
                if (id_a == id_b) {
                    TP++;
                    // 输出TP到文件
                    fout_TP << id_a << "," << id_b << "," << data.similarity << std::endl;
                    // // cout log
                    // std::cout << "TP" << std::endl;
                } else {
                    FP++;
                    fout_FP << id_a << "," << id_b << "," << data.similarity << std::endl;

                }
            } else {
                std::string id_a = data.id_a.substr(0, data.id_a.find('_'));
                std::string id_b = data.id_b.substr(0, data.id_b.find('_'));
                if (id_a == id_b) {
                    FN++;
                    // 输出FN到文件
                    fout_FN << id_a << "," << id_b << "," << data.similarity << std::endl;
                } else {
                    TN++;
                    fout_TN << id_a << "," << id_b << "," << data.similarity << std::endl;
                }
            }
            
        
        }

        

        // 计算评价指标并写入文件
        double TPR = static_cast<double>(TP) / (TP + FN);
        double FPR = static_cast<double>(FP) / (FP + TN);
        double TNR = static_cast<double>(TN) / (TN + FP);
        double FNR = static_cast<double>(FN) / (TP + FN);
        double ACC = static_cast<double>(TP + TN) / total;
        double Precision = static_cast<double>(TP) / (TP + FP);
        double Recall = static_cast<double>(TP) / (TP + FN);
        double F1 = 2 * Precision * Recall / (Precision + Recall);

        std::stringstream ss_r;
        ss_r << std::fixed << std::setprecision(2) << th;
        std::string th_str_r = ss_r.str();
        std::string result_file = result_dir + "/result_" + th_str_r + ".md";
        output_result(result_file, TP, TN, FP, FN, TPR, FPR, TNR, FNR, ACC, Precision, Recall, F1);
        fout_TN.close();
        fout_FN.close();
        fout_FP.close();
        fout_TP.close();
    }
    
    return 0;
}
