// # coding = utf-8
// # @Time    : 2023-11-01  09:05:05
// # @Author  : zhaosheng@nuaa.edu.cn
// # @Describe: Calc EER.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <getopt.h>
#include <cstring>

using namespace std;

struct Record {
    string speaker_a;
    string speaker_b;
    float similarity;
};

vector<Record> read_records(const char* filename) {
    vector<Record> records;
    ifstream ifs(filename);
    if (!ifs.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return records;
    }
    string line;
    while (getline(ifs, line)) {
        istringstream iss(line);
        Record record;
        getline(iss, record.speaker_a, ',');
        getline(iss, record.speaker_b, ',');
        iss >> record.similarity;
        records.push_back(record);
    }
    ifs.close();
    return records;
}

void compute_counts(float threshold, const vector<Record>& records, int& tp, int& fp, int& tn, int& fn, float& progress, float& threshold_processed, const string& save_path) {
    tp = fp = tn = fn = 0;
    progress = 0.0;
    threshold_processed = threshold;

    for (const auto& record : records) {
        string spkaid = record.speaker_a.substr(0, record.speaker_a.find('_'));
        string spkbid = record.speaker_b.substr(0, record.speaker_b.find('_'));
        bool positive = record.similarity >= threshold;

        if (spkaid == spkbid) {
            if (positive) {
                tp++;
            } else {
                fn++;
                // 写入 fn_<threshold>.txt 文件
                ofstream fout((save_path + "/fn_" + to_string(threshold) + ".txt").c_str(), ios::app);
                fout << record.speaker_a << ", " << record.speaker_b << ", " << record.similarity << endl;
                fout.close();
            }
        } else {
            if (positive) {
                fp++;
                // 写入 fp_<threshold>.txt 文件
                ofstream fout((save_path + "/fp_" + to_string(threshold) + ".txt").c_str(), ios::app);
                fout << record.speaker_a << ", " << record.speaker_b << ", " << record.similarity << endl;
                fout.close();
            } else {
                tn++;
                // 写入 tn_<threshold>.txt 文件
                ofstream fout((save_path + "/tn_" + to_string(threshold) + ".txt").c_str(), ios::app);
                fout << record.speaker_a << ", " << record.speaker_b << ", " << record.similarity << endl;
                fout.close();
            }
        }

        progress = (float)(&record - &records[0]) / records.size();

        threshold_processed = threshold;

        // // 实时可视化进度条
        // cout << "Threshold: " << threshold_processed << ", Progress: [" << string(progress * 50, '*') << string(50 - progress * 50, '-') << "] " << (int)(progress * 100) << "%" << '\r' << flush;
    }

    float precision = (float)tp / (tp + fp);
    float recall = (float)tp / (tp + fn);
    printf("\nThreshold:%.6f\tPrecision:%.6f\tRecall:%.6f\n", threshold_processed, precision, recall);
}

struct Metrics {
    float precision;
    float recall;
    float accuracy;
};

vector<Metrics> compute_metrics(const vector<Record>& records, const float start_threshold, const float end_threshold, const float step, const string& save_path) {
    cout << "Calculating metrics..." << endl;
    vector<Metrics> metrics;
    int total_steps = (int)((end_threshold - start_threshold) / step) + 1;
    int current_step = 0;
    float progress;
    float threshold_processed;
    for (float threshold = start_threshold; threshold <= end_threshold; threshold += step) {
        int tp, fp, tn, fn;
        compute_counts(threshold, records, tp, fp, tn, fn, progress, threshold_processed, save_path);
        Metrics m;
        m.precision = (float)tp / (tp + fp);
        m.recall = (float)tp / (tp + fn);
        m.accuracy = (float)(tp + tn) / records.size();
        metrics.push_back(m);
        current_step++;
        cout << "Threshold: " << threshold_processed << ", Progress: [" << string(progress * 50, '*') << string(50 - progress * 50, '-') << "] " << (int)(progress * 100) << "%" << '\r' << flush;
    }
    cout << endl;

    // 计算 Equal Error Rate
    float eer_threshold;
    float eer_diff = 100.0; // initial value
    for (const auto& m : metrics) {
        float diff = fabs(m.precision - (1 - m.recall));
        if (diff < eer_diff) {
            eer_diff = diff;
            eer_threshold = m.accuracy;
        }
    }
    printf("Equal Error Rate: %.6f\n", eer_threshold);

    // 计算 minDCF
    float p_target = 0.0001; // 目标 P_target
    float c_miss = 1.0; // 计算 C_miss
    float c_fa = 10.0; // 计算 C_fa
    float minDCF = 100.0; // initial value
    for (const auto& m : metrics) {
        float dcf = p_target * m.recall * c_miss + (1 - p_target) * (1 - m.precision) * c_fa;
        if (dcf < minDCF) {
            minDCF = dcf;
        }
    }
    printf("minDCF: %.6f\n", minDCF);

    // save minDCF and EER and table of metrics to result.txt with markdown format table.
    ofstream fout((save_path + "/result.txt").c_str(), ios::app);
    fout << "| Threshold | Precision | Recall | Accuracy |" << endl;
    fout << "| --------- | --------- | ------ | -------- |" << endl;
    for (const auto& m : metrics) {
        fout << "| " << m.accuracy << " | " << m.precision << " | " << m.recall << " | " << m.accuracy << " |" << endl;
    }
    fout << endl;
    fout << "Equal Error Rate: " << eer_threshold << endl;
    fout << "minDCF: " << minDCF << endl;
    fout << "Total Test Samples: " << records.size() << endl;


    return metrics;
}

void plot_curve(const vector<Metrics>& metrics) {
    FILE* gp = popen("gnuplot", "w");
    fprintf(gp, "set xlabel 'False Positive Rate'\n");
    fprintf(gp, "set ylabel 'True Positive Rate'\n");
    fprintf(gp, "set xrange [0:1]\n");
    fprintf(gp, "set yrange [0:1]\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set title 'Equal Error Rate Curve'\n");
    fprintf(gp, "set key top left box\n");
    fprintf(gp, "plot '-' using 2:3 with lines title 'EER Curve'\n");
    for (const auto& m : metrics) {
        fprintf(gp, "%.6f %.6f %.6f\n", m.accuracy, 1 - m.recall, m.precision);
    }
    fprintf(gp, "e\n");

    if (gp != NULL) {
        if (system("which xdg-open > /dev/null") == 0) {
            fprintf(gp, "set terminal png\n");
            fprintf(gp, "set output 'eer_curve.png'\n");
            fprintf(gp, "replot\n");
            fprintf(gp, "set terminal x11\n");
            fprintf(gp, "pause -1\n");
        } else {
            cerr << "Unable to save EER curve as image, please install xdg-open." << endl;
            fprintf(gp, "pause -1\n");
        }
    } else {
        cerr << "Failed to open gnuplot, please install gnuplot first." << endl;
    }
    pclose(gp);
}



int main(int argc, char* argv[]) {
    char* filename = "result.txt";
    float start_threshold = 0.0;
    float end_threshold = 1.0;
    float step = 0.01;
    bool plot = false;
    string save_path = "./"; // 默认路径为当前路径

    int opt;
    
    // 解析其它命令行参数
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            string arg_name = string(argv[i]);
            string arg_value = string(argv[i + 1]);

            if (arg_name == "--input") {
                // change arg_value to filename format
                // conver string  arg_value to char* filename_n
                char* filename_n = new char[arg_value.length() + 1];
                strcpy(filename_n, arg_value.c_str());
                filename = argv[i + 1];
            } else if (arg_name == "--start-threshold") {
                start_threshold = stof(arg_value);
            } else if (arg_name == "--end-threshold") {
                end_threshold = stof(arg_value);
            } else if (arg_name == "--step") {
                step = stof(arg_value);
            } else if (arg_name == "--plot") {
                // plot = true;
                if (arg_value == "1") {
                    plot = true;
                } else if (arg_value == "0") {
                    plot = false;
                } else {
                    cerr << "Invalid value for --plot: " << arg_value << endl;
                    return -1;
                }
            } else if (arg_name == "--savepath") {
                save_path = arg_value;
                cout << save_path << endl;
            } else if (arg_name == "--help") {
                cout << "Usage: " << argv[0] << " --start-threshold value --end-threshold value --step value --plot 1 --savepath path" << endl;
                return 0;
            }
            
        }
    }
    cout << "Now save path is: " << save_path << endl;
    // mkdir save_path if not exists
    if (system(("mkdir -p " + save_path).c_str()) != 0) {
        cerr << "Failed to create directory: " << save_path << endl;
        return -1;
    }

    auto start_time = chrono::steady_clock::now();
    vector<Record> records = read_records(filename);
    auto end_time = chrono::steady_clock::now();
    cout << "Read file in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;

    start_time = chrono::steady_clock::now();
    vector<Metrics> metrics = compute_metrics(records, start_threshold, end_threshold, step, save_path);
    end_time = chrono::steady_clock::now();
    cout << "Compute metrics in " << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << "s" << endl;

    if (plot) {
        plot_curve(metrics);
    }

    return 0;
}