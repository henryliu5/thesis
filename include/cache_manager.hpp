#include <boost/lockfree/spsc_queue.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread.hpp>
#include <queue>
#include <thread>
#include <torch/torch.h>
#include <unordered_map>
#include <functional>
#include <pthread.h>
#include <queue>
#include "concurrentqueue.hpp"


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

template<typename Data>
class concurrent_queue
{
private:
    std::queue<Data> the_queue;
    mutable boost::mutex the_mutex;
    boost::condition_variable the_condition_variable;
    bool alive = true;
public:
    void disable(){
        alive = false;
        the_condition_variable.notify_all();
    }

    void push(Data const& data)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        the_queue.push(data);
        lock.unlock();
        the_condition_variable.notify_one();
    }

    bool empty() const
    {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_queue.empty();
    }

    bool try_pop(Data& popped_value)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        if(the_queue.empty())
        {
            return false;
        }
        
        popped_value=the_queue.front();
        the_queue.pop();
        return true;
    }

    bool wait_and_pop(Data& popped_value)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while(the_queue.empty())
        {
            the_condition_variable.wait(lock);
            if(!alive){
                return false;
            }
        }
        
        popped_value=the_queue.front();
        the_queue.pop();
        return true;
    }

};

class CacheManager {
    /** Controls cache state.
     *  Computes cache usage statistics and performs dynamic cache updates.
     */
private:
    int cache_size;
    int update_frequency;
    int decay_frequency;
    int staging_area_size;

    // TODO switch to Boost lockfree queue if multiple producers
    // boost::lockfree::spsc_queue<torch::Tensor, boost::lockfree::capacity<1024>> q;
    // moodycamel::ConcurrentQueue<torch::Tensor> q;
    concurrent_queue<torch::Tensor> q;

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
    torch::Tensor counts;
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
        //!! Need to make sure worker is alive in this loop!
        while (finished_threads.load() < fetchers_at_start && worker_alive)
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
        pthread_setname_np(pthread_self(), "CacheManager worker");
        // Starts at 1 just to skip update
        int requests_handled = 1;
        while (worker_alive) {
            bool decay = false;
            bool update = false;

            torch::Tensor nids;
            if(q.wait_and_pop(nids)){

                auto counts_acc = counts.accessor<long, 1>();
                auto nids_acc = nids.accessor<long, 1>();
                int n = nids.sizes()[0];
                for(int i = 0; i < n; i++){
                    counts_acc[nids_acc[i]] += 1; 
                }

                requests_handled++;
                // cout << "handled: " << requests_handled << "\n";
                if(decay_frequency != 0 && requests_handled % decay_frequency == 0){
                    decay = true;
                }                
                if(update_frequency != 0 && requests_handled % update_frequency == 0) {
                    update = true;
                }
            }
            // while (!q.empty() && worker_alive) {
            //     // q.consume_all<null_fn>();
            //     // torch::Tensor nids = q.front();
            //     // // Probably should go after the q.pop but now you can wait for
            //     // // the sum to finish
            //     // if(nids.sizes()[0] > 0){
            //     //     ASSERT ((nids.max().item<long>() < counts.sizes()[0]), "Invalid node id received at manager " << nids.max());
            //     //     ASSERT ((nids.min().item<long>() >= 0), "Invalid node id received at manager " << nids.min());
            //     //     // std::cout << nids << std::endl;
                    
            //     //     // torch::Tensor nid_cur_counts = counts.index({nids});
            //     //     // // TODO - assumes nids are unique, can use torch::bincount if not
            //     //     // torch::Tensor new_nids_counts = nid_cur_counts + 1;

            //     //     // std::cout << "nids dtype: " << nids.dtype() <<endl;
            //     //     // std::cout << "new_nids_counts dtype: " << new_nids_counts.dtype() << endl;
            //     //     // cout << "nids " << nids.sizes()[0] << " " << nids.device() << endl;
            //     //     // cout << "new nid counts " << new_nids_counts.sizes()[0] << " " << new_nids_counts.device() << endl;
            //     //     // cout << "counts " << counts.sizes()[0] << " " << counts.device() << endl;
            //     //     // // cout << nids << endl;
            //     //     // std::cout << "counts size: " << counts.sizes()[0] << endl;
            //     //     // cout << "max 2: " << nids.max().item<long>() << endl;
            //     //     // cout << "max 2: " << nids.min().item<long>() << endl;

            //     //     // counts.index_put_({nids}, new_nids_counts);

            //     //     // torch::Tensor hist = nids.bincount({}, counts.sizes()[0]);
            //     //     // cout << hist.sizes()[0] << endl;
            //     //     // counts += hist;
            //     //     // std::cout << "new" << counts << std::endl;

            //     // }
            //     q.pop();
            //     requests_handled++;
            //     // cout << "handled: " << requests_handled << "\n";
            //     if(decay_frequency != 0 && requests_handled % decay_frequency == 0){
            //         decay = true;
            //     }                
            //     if(update_frequency != 0 && requests_handled % update_frequency == 0) {
            //         update = true;
            //     }
            // }

            if(decay){
                // cout << "decaying counts" << endl;
                // counts.div_(2, "floor");
                auto counts_acc = counts.accessor<long, 1>();
                int n = counts.sizes()[0];
                for(int i = 0; i < n; i++){
                    long c = counts_acc[i];
                    if (c != 0){
                        counts_acc[i] = c / 2;
                    }
                    
                }
                // cout << "counts: " << counts << endl;
            }

            // Cache update
            if(update) {
                // cout << "starting stats" << endl;
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
                cout << "replacement size: " << i1 << endl;
                
                auto current_worst_avg_count = counts.index({replace_nids}).mean(torch::kFloat32).item();
                auto replace_avg_count = counts.index({add_nids}).mean(torch::kFloat32).item();
                cout << "replacing " << current_worst_avg_count << " with " << replace_avg_count << endl;

                ASSERT (add_nids.sizes()[0] == replace_nids.sizes()[0], "Internal error, add/replace mismatch " << add_nids.sizes()[0] << " " << replace_nids.sizes()[0] << endl);

                // Gather and move features to GPU
                cacheUpdate(add_nids, replace_nids);
            }
        }
        std::cout << "worker exited" << std::endl;
        
    }


public:
    CacheManager(const int num_total_nodes, const int cache_size, const int update_frequency, const int decay_frequency, const int staging_area_size)
        : cache_size(cache_size)
        , worker_alive(true)
        , worker_thread(&CacheManager::worker, this)
        , update_frequency(update_frequency)
        , decay_frequency(decay_frequency)
        , staging_area_size(staging_area_size)
        , started_threads(0)
        , finished_threads(0)
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
            q.disable();

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
        // q.enqueue(node_ids);
    }

    void waitForQueue()
    {
        gilRelease([this]{
            // while (!q.empty())
                // ;
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
        // cout << "finished: " << finished_threads << "\n";
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
