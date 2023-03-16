#include <iostream>
#include <queue>
#include <thread>
#include "cache_manager.hpp"

using std::cout, std::endl;

void CacheManager::gilRelease(std::function<void()> f) {
    f();
}

int main(){
    // int num_nodes = 10;
    // auto cache_mask = torch::zeros(num_nodes, torch::dtype(torch::kBool));
    // auto cache_mapping = torch::zeros(num_nodes, torch::device(torch::kCUDA));
    // std::unordered_map<std::string, torch::Tensor> cache;

    // CacheManager c(num_nodes, 10, 0, 0);//, cache_mask, cache_mapping, cache);

    // int neighborhood = 5;
    // torch::Tensor rand_nids = torch::randint(num_nodes, {neighborhood, });

    // c.incrementCounts(rand_nids);

    // c.waitForQueue();
    // cout << c.getCounts() << endl;

    // torch::Tensor foo = torch::rand({100});

    // auto [val, idx] = foo.topk(100);
    // // // assert foo is 2-dimensional and holds floats.
    // // auto foo_a = foo.accessor<float,2>();
    // // float trace = 0;

    // // for(int i = 0; i < foo_a.size(0); i++) {
    // // // use the accessor foo_a to get tensor data.
    // // trace += foo_a[i][i];
    // // }

    // // torch::Tensor add_nids = torch::ones(2, torch::dtype(torch::kLong));
    // torch::Tensor add_nids = idx;
    // cout << add_nids.has_storage() << endl;
    // // cout << add_nids.dtype_initialized() << endl;
    // auto add_nids_acc = add_nids.accessor<long, 1>();

    torch::Tensor nids = torch::randint(100, {0});
    torch::Tensor graph_features = torch::rand({100, 100});
    auto cpu_staging_area = graph_features.index({nids}).to(torch::device(torch::kCPU).pinned_memory(true));
    auto gpu_staging_area = cpu_staging_area.to(torch::device(torch::kCUDA), true);

    torch::Tensor cache_idxs = torch::zeros(0);
    torch::Tensor cache = torch::zeros(100);
    cache.index_put_({cache_idxs}, gpu_staging_area);
}