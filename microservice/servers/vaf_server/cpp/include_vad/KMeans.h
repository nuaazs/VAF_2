#ifndef KMEANS_H
#define KMEANS_H
#define DEBUG
#include <vector>

class KMeans {
public:
    enum DistanceType {
        EUCLIDEAN,
        COSINE
    };

    static std::vector<int> run(const std::vector<std::vector<double>>& data, int n, int k_min, int k_max, int max_iter, double discard_threshold, DistanceType distance_type);
    static double calculateDistance(const std::vector<double>& v1, const std::vector<double>& v2, DistanceType distance_type);
private:
    static int findNearestCluster(const std::vector<double>& v, const std::vector<std::vector<double>>& centroids, DistanceType distance_type);
    
};

#endif  // KMEANS_H
