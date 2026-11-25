#include "print_stats.h"


void print_stats(std::vector<double> time_data){
    std::sort(time_data.begin(),time_data.end());
    
    double sum = std::accumulate(time_data.begin(), time_data.end(), 0.0);
    double mean = sum / time_data.size();

    double p50 = time_data[time_data.size() * 0.5];
    double p95 = time_data[time_data.size() * 0.95];
    double p99 = time_data[time_data.size() * 0.99];   

    double min = time_data.front();
    double max = time_data.back();

    std::cout<<"----Summary----"<<"\n";
    std::cout<<"Max inference time : "<<max<<"\n";
    std::cout<<"Min inference time : "<<min<<"\n";
    std::cout<<"Mean inference time : "<<mean<<"\n";
    std::cout<<"p50 : "<<p50<<"\n";
    std::cout<<"p95 : "<<p95<<"\n";
    std::cout<<"p99 : "<<p99<<"\n";
}