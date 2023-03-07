#include <iostream>
#include <queue>
#include <thread>

class CacheManager {
private:
    int counts;
    int cache_size;
    // TODO switch to Boost lockfree queue if multiple producers
    std::queue<int> q;

    //!! The order here is important since worker_alive must be intialized first
    volatile bool worker_alive;
    std::thread worker_thread;

public:
    CacheManager (const int num_total_nodes, const int cache_size)
        : cache_size(cache_size),
          worker_alive(true),
          worker_thread(&CacheManager::worker, this)
    {
        counts = 0;
    }

    ~CacheManager(){
        std::cout << "entered destructor" << std::endl;
        // while(!q.empty()); // This waits for worker to finish processing
        worker_alive = false;
        worker_thread.join();
    }

    void receiveCounts(int node_ids){
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

int main(){
    CacheManager c(100, 10);

    int rand_nids = 10;

    c.receiveCounts(rand_nids);

}