#ifndef BenchMarkPCG_H
#define BenchMarkPCG_H

#include <numeric>
#include <iostream>
#include <cstdio>
#include <vector>

//The struct holds each operation benchmarks
struct BenchMarkPCG {
    std::vector<double> q_ap_times;
    std::vector<double> alpha_times;
    std::vector<double> x_update_times;
    std::vector<double> r_update_times;
    std::vector<double> s_update_times;
    std::vector<double> beta_times;
    std::vector<double> d_update_times;
};

//Process: the method erases the first result from the vector 
void eliminateFirst(BenchMarkPCG& bmPCG){
    bmPCG.q_ap_times.erase(bmPCG.q_ap_times.begin());
    bmPCG.alpha_times.erase(bmPCG.alpha_times.begin());
    bmPCG.x_update_times.erase(bmPCG.x_update_times.begin());
    bmPCG.r_update_times.erase(bmPCG.r_update_times.begin());
    bmPCG.s_update_times.erase(bmPCG.s_update_times.begin());
    bmPCG.beta_times.erase(bmPCG.beta_times.begin());
    bmPCG.d_update_times.erase(bmPCG.d_update_times.begin());
}

double calculateAveratges(const std::vector<double>& vals){
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    return sum / vals.size();
}

//Process: getting average times from each vector
std::vector<double> getAverages(BenchMarkPCG& bmPCG){
    std::vector<double> averages;
    averages.push_back(calculateAveratges(bmPCG.q_ap_times));
    averages.push_back(calculateAveratges(bmPCG.alpha_times));
    averages.push_back(calculateAveratges(bmPCG.x_update_times));
    averages.push_back(calculateAveratges(bmPCG.r_update_times));
    averages.push_back(calculateAveratges(bmPCG.s_update_times));
    averages.push_back(calculateAveratges(bmPCG.beta_times));
    averages.push_back(calculateAveratges(bmPCG.d_update_times));
    return averages;   
}



//Provess: Print out the each average value with corresponding operation name
void printResult(const std::vector<double>& averages){
    const char* labels[] = {
        "q <- Ad",
        "alpha <- delta_{new} / d^{T} * q",
        "x_{i+1} <- x_{i} + alpha * d",
        "r_{i+1} <- r_{i} - alpha * q",
        "s <- M * r",
        "beta <- r' * s / delta_old",
        "d_{i+1} <- s + d_{i} * beta"
    };

    for(size_t i = 0; i < averages.size(); i++){
        printf("%s: %f s\n", labels[i], averages[i]);
    }
}

void clearBenchMarkPCG(BenchMarkPCG& bmPCG){
        bmPCG.q_ap_times.clear();
        bmPCG.alpha_times.clear();
        bmPCG.x_update_times.clear();
        bmPCG.r_update_times.clear();
        bmPCG.s_update_times.clear();
        bmPCG.beta_times.clear();
        bmPCG.d_update_times.clear();
}

#endif //BenchMarkPCG_H