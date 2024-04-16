#!/bin/bash

# Build top1_multi_model
echo "Build ../bin/top1_multi_model"
echo "    *-> Used for Multi Model"
# g++ get_top1_and_score.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o ../bin/top1_multi_model
g++ get_top1_and_score_multi_model.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o ../bin/top1_multi_model

# Build top1acc
echo "Build ../bin/top1acc"
echo "    *-> Used for Calc ACC result from Multi Model"
g++ get_top1_acc.cpp -std=c++17 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o ../bin/top1acc

# Build eer
echo "Build ../bin/eer"
echo "    *-> Used for Calc EER result from Multi Model"
g++ get_eer.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o ../bin/eer
