#include <iostream>
#include <queue>
#include <thread>
#include "cache_manager.hpp"

using std::cout, std::endl;

void CacheManager::gilRelease(std::function<void()> f) {
    f();
}

int main(){
    int num_nodes = 10;
    auto cache_mask = torch::zeros(num_nodes, torch::dtype(torch::kBool));
    auto cache_mapping = torch::zeros(num_nodes, torch::device(torch::kCUDA));
    std::unordered_map<std::string, torch::Tensor> cache;

    CacheManager c(num_nodes, 10, 0, 0);//, cache_mask, cache_mapping, cache);

    int neighborhood = 5;
    torch::Tensor rand_nids = torch::randint(num_nodes, {neighborhood, });

    c.incrementCounts(rand_nids);

    c.waitForQueue();
    cout << c.getCounts() << endl;
}