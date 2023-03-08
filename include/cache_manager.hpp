#include <boost/lockfree/spsc_queue.hpp>
#include <queue>
#include <thread>
#include <torch/torch.h>
#include <unordered_map>
#include <functional>

#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

using torch::indexing::Slice, torch::indexing::None;
using std::cout, std::endl;

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

    // References to cache-related data structures seen in Python
    torch::Tensor graph_features;
    torch::Tensor cache_mask;
    torch::Tensor cache_mapping;
    torch::Tensor reverse_mapping;
    // std::unordered_map<std::string, torch::Tensor> cache;
    torch::Tensor cache;

    // CacheManager specific cache metadata
    std::atomic<long> started_threads;
    std::atomic<long> finished_threads;
    torch::Tensor cpu_staging_area;
    torch::Tensor gpu_staging_area;

    void stageNewFeatures(torch::Tensor nids)
    {
        // Feature gather
        // TODO figure out how to make this just go directly into the pinned mem buf
        cpu_staging_area = graph_features.index({nids}).to(torch::device(torch::kCPU).pinned_memory(true));
        gpu_staging_area = cpu_staging_area.to(torch::device(torch::kCUDA), true);
    }

    /**
     * Add new_nids into the cache, replacing entries correpsonding to replace_nids.
     * Performs operations such that cache readers only ever interact with consistent
     * cache entries.
     * "Consistent cache entry" -> feature at cache[cache_mapping[i]] == feature for node i
    */
    void cacheUpdate(torch::Tensor new_nids, torch::Tensor replace_nids)
    {
        stageNewFeatures(new_nids);
        
        // TODO could do all of this in a callback, thus happening async from counting
        torch::Tensor replace_cache_idxs = cache_mapping.index({replace_nids});
        // 1. Mask off cache
        cache_mask.index_put_({ replace_nids }, false);

        // 2. Wait for enough threads to finish
        long fetchers_at_start = started_threads.load();
        while (finished_threads.load() < fetchers_at_start)
            ;

        // 3a. Perform remappings
        cache_mapping.index_put_({ replace_nids }, -1);
        reverse_mapping.index_put_({ replace_cache_idxs }, new_nids);
        cache_mapping.index_put_({ new_nids }, replace_cache_idxs);

        // 3b. Actually update the cache
        // cout << "replace_cache_idxs " << replace_cache_idxs.sizes()[0] << endl;
        // cout << "gpu_staging_area " << gpu_staging_area.sizes()[0] << endl;
        // cout << "cache dev: " << cache.device() << endl;
        // cout << "gpu staging area dev: " << gpu_staging_area.device() << endl;
        cache.index_put_({ replace_cache_idxs }, gpu_staging_area);

        // 4. Unmask, but now with new nodes!
        cache_mask.index_put_({ new_nids }, true);
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
                cout << "starting stats" << endl;
                // Compute cache statistic
                torch::Tensor add_nids = getMostCommonNodesNotInCache(staging_area_size);
                torch::Tensor replace_cache_idxs = getLeastUsedCacheIndices(staging_area_size);
                torch::Tensor replace_nids = reverse_mapping.index({replace_cache_idxs});

                // Dumb way to figure out which to replace
                int n = staging_area_size;
                int i1 = 0;
                int i2 = 0;
                auto add_nids_acc = add_nids.accessor<long, 1>();
                auto replace_nids_acc = replace_nids.accessor<long, 1>();
                auto counts_acc = counts.accessor<long, 1>();
                while(i1 + i2 < n){
                    if(counts_acc[add_nids_acc[i1]] > counts_acc[replace_nids_acc[i2]]){
                        i1 += 1;
                    } else {
                        i2 += 1;
                    }
                }

                add_nids = add_nids.index({Slice(None, i1)});
                replace_nids = replace_nids.index({Slice(i2, None)});

                ASSERT (add_nids.sizes()[0] == replace_nids.sizes()[0], "Internal error, add/replace mismatch " << add_nids.sizes()[0] << " " << replace_nids.sizes()[0] << endl);

                // Gather and move features to GPU
                cacheUpdate(add_nids, replace_nids);
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
        ASSERT (staging_area_size <= cache_size, "staging_area_size must be smaller than the cache size, staging_area_size: " << staging_area_size << " cache_size: " << cache_size);
        counts = torch::zeros(num_total_nodes, torch::dtype(torch::kLong));
        cpu_staging_area = torch::empty(staging_area_size, torch::device(torch::kCPU).pinned_memory(true));
        gpu_staging_area = torch::empty(staging_area_size, torch::device(torch::kCUDA).requires_grad(false));
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
        this->graph_features = graph_features;
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
       torch::Tensor counts_in_cache = counts.masked_select(cache_mask);
       ASSERT (k <= cache_size, "k must be smaller than the cache size, k: " << k << " cache_size: " << cache_size);
       auto [values, indices] = counts_in_cache.topk(k, -1, false);
       return indices;
    }

    torch::Tensor getCounts()
    {
        return counts;
    }

    void gilRelease(std::function<void()> f);
};
