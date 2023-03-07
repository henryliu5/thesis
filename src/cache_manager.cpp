#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <queue>
#include <thread>


class CacheManager {
    /** Controls cache state. 
     *  Computes cache usage statistics and performs dynamic cache updates.
    */
private:
    torch::Tensor counts;
    int cache_size;
    // TODO switch to Boost lockfree queue if multiple producers
    std::queue<torch::Tensor> q;

    //!! The order here is important since worker_alive must be intialized first
    volatile bool worker_alive;
    std::thread worker_thread;

public:
    CacheManager (const int num_total_nodes, const int cache_size)
        : cache_size(cache_size),
          worker_alive(true),
          worker_thread(&CacheManager::worker, this)
    {
        counts = torch::zeros(num_total_nodes);
    }

    ~CacheManager(){
        std::cout << "entered destructor" << std::endl;
        // while(!q.empty()); // This waits for worker to finish processing
        worker_alive = false;
        std::cout << "bool set, calling join" << std::endl;
        worker_thread.join();
        std::cout << "exiting destructor" << std::endl;
    }

    void receiveCounts(torch::Tensor node_ids){
        q.push(node_ids);
    }

    void worker(){
        while(worker_alive){
            if(!q.empty()){
                auto nids = q.front();
                q.pop();
                counts += nids;
                std::cout << "new counts: " << counts << std::endl;
            }
        }
        std::cout << "worker exited" << std::endl;
    }
};


PYBIND11_MODULE(fast_inference_cpp, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<CacheManager>(m, "CacheManager")
        .def(py::init<const int, const int>())
        .def("receive_counts", &CacheManager::receiveCounts);
}
