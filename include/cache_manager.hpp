#include <boost/lockfree/spsc_queue.hpp>
#include <queue>
#include <thread>
#include <torch/torch.h>
#include <unordered_map>
#include <functional>

class CacheManager {
    /** Controls cache state.
     *  Computes cache usage statistics and performs dynamic cache updates.
     */
private:
    torch::Tensor counts;
    int cache_size;
    int update_frequency;
    int staging_area_size;

    // TODO switch to Boost lockfree queue if multiple producers
    boost::lockfree::spsc_queue<torch::Tensor, boost::lockfree::capacity<1024>> q;

    //!! The order here is important since worker_alive must be intialized first
    volatile bool worker_alive;
    std::thread worker_thread;

    // References to cache data structures
    torch::Tensor cache_mask;
    torch::Tensor cache_mapping;
    torch::Tensor reverse_mapping;
    // std::unordered_map<std::string, torch::Tensor> cache;
    torch::Tensor cache;

    std::atomic<long> started_threads;
    std::atomic<long> finished_threads;

    void stageNewFeatures(torch::Tensor nids)
    {
    }

    /**
     * Add new_nids into the cache, replacing entries correpsonding to replace_nids.
     * Performs operations such that cache readers only ever interact with consistent
     * cache entries.
     * "Consistent cache entry" -> feature at cache[cache_mapping[i]] == feature for node i
    */
    void cacheUpdate(torch::Tensor new_nids, torch::Tensor replace_nids)
    {
        // 1. Mask off cache
        cache_mapping.index_put_({ replace_nids }, false);

        // 2. Wait for enough threads to finish
        long fetchers_at_start = started_threads.load();
        while (finished_threads.load() < fetchers_at_start)
            ;

        // 3. Perform remappping and update
    }

    void worker()
    {
        // Starts at 1 just to skip update
        int requests_handled = 1;
        while (worker_alive) {
            if (!q.empty()) {
                torch::Tensor nids = q.front();
                // std::cout << nids << std::endl;
                // Probably should go after the q.pop but now you can wait for
                torch::Tensor nid_cur_counts = counts.index({nids});
                // std::cout << nid_cur_counts << std::endl;
                
                // TODO - assumes nids are unique, can use torch::bincount if not
                counts.index_put_({nids}, nid_cur_counts + 1);
                // std::cout << "new" << counts << std::endl;
                q.pop();
                requests_handled++;
            }

            // Cache update
            if(update_frequency != 0 && requests_handled % update_frequency == 0) {
                // Compute cache statistic
                torch::Tensor add_nids = getMostCommonNodesNotInCache(staging_area_size);
                torch::Tensor replace_cache_idxs = getLeastUsedCacheIndices(staging_area_size);
                torch::Tensor replace_nids = reverse_mapping.index({replace_cache_idxs});

                // Start movement of those features to GPU
            }
        }
        std::cout << "worker exited" << std::endl;
        
    }


public:
    CacheManager(const int num_total_nodes, const int cache_size, const int update_frequency, const int staging_area_size)
        : cache_size(cache_size)
        , worker_alive(true)
        , worker_thread(&CacheManager::worker, this)
        , update_frequency(update_frequency)
        , staging_area_size(staging_area_size)
    {
        counts = torch::zeros(num_total_nodes);
    }

    ~CacheManager()
    {
        gilRelease([this](){
            std::cout << "entered destructor" << std::endl;
            // while(!q.empty()); // This waits for worker to finish processing
            worker_alive = false;
            std::cout << "bool set, calling join" << std::endl;
            worker_thread.join();
            std::cout << "exiting destructor" << std::endl;
        }
        );
    }

    void setCache(torch::Tensor graph_features, torch::Tensor cache_mask, torch::Tensor cache_mapping,
        torch::Tensor reverse_mapping, torch::Tensor cache)
    {

        this->cache_mask = cache_mask;
        this->cache_mapping = cache_mapping;
        this->reverse_mapping = reverse_mapping;
        this->cache = cache;
    }

    void setUpdateFrequency(int update_frequency)
    {
        this->update_frequency = update_frequency;
    }

    void setStagingAreaSize(int size)
    {
        this->staging_area_size = size;
    }

    void incrementCounts(torch::Tensor node_ids)
    {
        q.push(node_ids);
    }

    void waitForQueue()
    {
        gilRelease([this]{
            while (!q.empty())
                ;
            }
        );
    }

    // TODO maybe rename to something like "Cache Fetchers"
    void threadEnter()
    {
        started_threads++;
    }

    void threadExit()
    {
        finished_threads++;
    }

    torch::Tensor getMostCommonNodesNotInCache(int k){
        /**
         * Returns most common nids not in cache based on current counts.
        */
       auto [values, indices] = counts.topk(cache_size + k);

       // topk_mask has shape indices.shape (cache_size + k), tells us if node in cache or not
       torch::Tensor topk_mask = cache_mask.index({indices});
       return indices.masked_select(topk_mask.logical_not());
    }

    torch::Tensor getLeastUsedCacheIndices(int k){
        /**
         * Returns least used cache indices based on current counts.
        */
       auto [values, indices] = counts.masked_select(cache_mask).topk(k, -1, false);
       return indices;
    }

    torch::Tensor getCounts()
    {
        return counts;
    }

    void gilRelease(std::function<void()> f);
};
