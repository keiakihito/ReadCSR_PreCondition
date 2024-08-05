#ifndef BenchMarkBFBCG_H
#define BenchMarkBFBCG_H

#include <numeric>
#include <iostream>
#include<cstdio>
#include <vector>

//The struct holds each operation benchmarks
struct BenchMarkBFBCG {
    std::vector<double> q_ap_times;
    std::vector<double> alpha_times;
    std::vector<double> x_update_times;
    std::vector<double> r_update_times;
    std::vector<double> z_update_times;
    std::vector<double> beta_times;
    std::vector<double> p_update_times;
};

//Process: the method erases the first result from the vector 
void eliminateFirst(BenchMarkBFBCG& bmBFBCG){
    bmBFBCG.q_ap_times.erase(bmBFBCG.q_ap_times.begin());
    bmBFBCG.alpha_times.erase(bmBFBCG.alpha_times.begin());
    bmBFBCG.x_update_times.erase(bmBFBCG.x_update_times.begin());
    bmBFBCG.r_update_times.erase(bmBFBCG.r_update_times.begin());
    bmBFBCG.z_update_times.erase(bmBFBCG.z_update_times.begin());
    bmBFBCG.beta_times.erase(bmBFBCG.beta_times.begin());
    bmBFBCG.p_update_times.erase(bmBFBCG.p_update_times.begin());
}

double calculateAveratges(const std::vector<double>& vals){
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    return sum / vals.size();
}

//Process: getting average times from each vector
std::vector<double> getAverages(BenchMarkBFBCG& bmBFBCG){
    std::vector<double> averages;
    averages.push_back(calculateAveratges(bmBFBCG.q_ap_times));
    averages.push_back(calculateAveratges(bmBFBCG.alpha_times));
    averages.push_back(calculateAveratges(bmBFBCG.x_update_times));
    averages.push_back(calculateAveratges(bmBFBCG.r_update_times));
    averages.push_back(calculateAveratges(bmBFBCG.z_update_times));
    averages.push_back(calculateAveratges(bmBFBCG.beta_times));
    averages.push_back(calculateAveratges(bmBFBCG.p_update_times));
    return averages;   
}



//Provess: Print out the each average value with corresponding operation name
void printResult(const std::vector<double>& averages){
    const char* labels[] = {
        "Q <- AP",
        "Alpha <- (P'Q)^{-1} * (P'R)",
        "X_{i+1} <- x_{i} + P * alpha",
        "R_{i+1} <- R_{i} - Q * alpha",
        "Z_{i+1} <- MR_{i+1}",
        "beta <- -(P'Q)^{-1} * (Q'Z_{i+1})",
        "P_{i+1} = orth(Z_{i+1} + p * beta)"
    };

    for(size_t i = 0; i < averages.size(); i++){
        printf("%s: %f s\n", labels[i], averages[i]);
    }
}

void clearBenchMarkBFBCG(BenchMarkBFBCG& bmBFBCG){
        bmBFBCG.q_ap_times.clear();
        bmBFBCG.alpha_times.clear();
        bmBFBCG.x_update_times.clear();
        bmBFBCG.r_update_times.clear();
        bmBFBCG.z_update_times.clear();
        bmBFBCG.beta_times.clear();
        bmBFBCG.p_update_times.clear();
}

#endif //BenchMarkBFBCG_H