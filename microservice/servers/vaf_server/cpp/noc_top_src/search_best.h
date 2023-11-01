#ifndef _SEARCHBEST_
#define _SEARCHBEST_

#include <assert.h>
#include <cmath>
#include <float.h>
#include <algorithm>
#include <climits>
#include <vector>
// use openblas
#include <cblas.h>
#include "cosine_similarity.h"

// Step 1, g++ main.cpp search_best.cpp cosine_similarity.cpp -std=c++11
// Step 2, g++ main.cpp search_best.cpp cosine_similarity.cpp -std=c++11 -O3
// Step 3, g++ main.cpp search_best.cpp cosine_similarity.cpp -std=c++11 -O3 -Ofast -ffast-math
// 定义返回类型，包含最相似的id(int)和相似度（float）
typedef struct {
    // int8 id;
    int id;
    float similarity;
} Result;

template <typename T>
std::vector<Result> SearchBest(const T* __restrict__ const pVecA,
                               const int lenA,
                               const T* __restrict__ const pVecDB,
                               const int lenDB,
                               const int topk)
{
    assert(lenDB % lenA == 0);
    const int featsize = lenA;
    const int facenum = lenDB / lenA;

    std::vector<Result> results;
    results.reserve(topk);

    for (int i = 0; i < facenum; i++) {
        T similarity = Cosine_similarity(pVecA, pVecDB + i * featsize, featsize);

        if (results.size() < topk) {
            results.push_back({i, similarity});
        } else {
            // 找到结果中最小的相似度
            float minSimilarity = FLT_MAX;
            int minIndex = -1;
            for (int j = 0; j < results.size(); j++) {
                if (results[j].similarity < minSimilarity) {
                    minSimilarity = results[j].similarity;
                    minIndex = j;
                }
            }

            // 如果当前相似度大于最小相似度，则替换掉最小相似度的结果
            if (similarity > minSimilarity) {
                results[minIndex] = {i, similarity};
            }
        }
    }

    return results;
}

template <typename T>
std::vector<Result> SearchMeanBest(const std::vector<const T*>& vecVecA,
                                   const int lenA,
                                   const std::vector<const T*>& vecDB,
                                   const int lenDB,
                                   const int topk)
{
    assert(vecVecA.size() == vecDB.size());
    assert(lenDB % lenA == 0);
    assert(lenDB % lenA == 0);
    const int featsize = lenA;
    const int facenum = lenDB / lenA;
    const int vecDBSize = vecDB.size();

    std::vector<float> simAll(facenum, 0.0f);
    std::vector<Result> results;
    results.reserve(topk);

    for (int j = 0; j < vecDBSize; ++j) {
        const T* pVecA = vecVecA[j];
        const T* pVecDB = vecDB[j];

        for (int i = 0; i < facenum; ++i) {
            T similarity = Cosine_similarity(pVecA, pVecDB + i * featsize, featsize);
            simAll[i] += similarity/vecDBSize;
        }
    }

    // 找到simAll中的k个最高结果
    // simAll 按照得分排序
    std::vector<int> indices(facenum);
    for (int i = 0; i < facenum; ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&simAll](int i1, int i2) { return simAll[i1] > simAll[i2]; });
    // 取前topk个
    for (int i = 0; i < topk; ++i) {
        results.push_back({indices[i], simAll[indices[i]]});
    }
    cout << "Top " << topk << " results:" << endl;
    cout << "id, similarity" << endl;
    cout << "----------------" << endl;
    for (int i = 0; i < topk; ++i) {
        cout << results[i].id << ", " << results[i].similarity << endl;
    }
    cout << "----------------" << endl;
    return results;
}

#endif //!_SEARCHBEST_
