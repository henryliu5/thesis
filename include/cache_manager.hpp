// #include <boost/lockfree/spsc_queue.hpp>
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
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
// #include "concurrentqueue.hpp"
#include <chrono>
#include <vector>
#include <algorithm>
using namespace std::chrono;


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

class AutoProfiler {
 public:
  AutoProfiler(std::string name)
      : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()) { }
  ~AutoProfiler() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() << " musec\n";
  }
 private:
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

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
    int BOUND = 5;
public:
    void disable(){
        alive = false;
        the_condition_variable.notify_all();
    }

    void push(Data const& data)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while(the_queue.size() >= BOUND){
            the_queue.pop();
        }
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

class TensorHolder {
    /** A place to put and get a single Tensor (or pair of Tensors)*/

    std::atomic<bool> empty;
    torch::Tensor new_gpu_features;
    torch::Tensor new_nids;
    int max_size;

public:
    TensorHolder(int max_size) : max_size(max_size) {}

    bool tryGet(torch::Tensor& feats, torch::Tensor& nids){
        if(empty){
            return false;
        }
        feats = new_gpu_features;
        nids = new_nids;
        empty.store(true);
        return true;
    }

    void setFeats(torch::Tensor feats, torch::Tensor nids){
        if(empty){
            new_gpu_features = feats.slice(0, 0, max_size);
            new_nids = nids.slice(0, 0, max_size);
            empty.store(false);
        }
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

    // Test variables
    torch::Tensor big_graph_arange;
    int requests_handled2;
    TensorHolder gpu_feat_holder;
    bool use_gpu_transfer;
    torch::Tensor topk_mask;

    void stageNewFeatures(torch::Tensor nids)
    {
        // Feature gather
        // TODO figure out how to make this just go directly into the pinned mem buf
        int n = nids.sizes()[0];
        // cout << "cpu_staging_area size: " << cpu_staging_area.sizes() << endl;
        // cout << "graph feat size: " << graph_features.sizes() << endl;
        // cout << "nids size: " << nids.sizes() << endl;

        cpu_staging_area.index_put_({Slice(0, n)}, graph_features.index({nids}));
        // cpu_staging_area = graph_features.index({nids}).to(torch::device(torch::kCPU).pinned_memory(true));

        // auto cpu_stage_acc = cpu_staging_area.accessor<float, 1>();
        // auto graph_acc = graph_features.accessor<float, 1>();
        // auto nids_acc = nids.accessor<long, 1>();
        // int n = nids.sizes()[0];
        // for(int i = 0; i < n; i++){
        //     cpu_stage_acc[i] = graph_acc[nids_acc[i]];
        // }

        gpu_staging_area = cpu_staging_area.index({Slice(0, n)}).to(torch::device(torch::kCUDA), true);
        // gpu_staging_area = cpu_staging_area.to(torch::device(torch::kCUDA), true);
    }

    /**
     * Add new_nids into the cache, replacing entries correpsonding to replace_nids.
     * Performs operations such that cache readers only ever interact with consistent
     * cache entries.
     * "Consistent cache entry" -> feature at cache[cache_mapping[i]] == feature for node i
    */
    void cacheUpdate(torch::Tensor new_nids, torch::Tensor replace_nids)
    {
        ASSERT(!use_gpu_transfer, "GPU transfer must be disabled for cache manager to initiate movement of new features to staging area");
        // auto start = high_resolution_clock::now();
        stageNewFeatures(new_nids);
        // auto stop = high_resolution_clock::now();
        // auto duration = duration_cast<microseconds>(stop - start);
        // cout << "staging time: " << duration.count() << endl;
        
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
        cache.index_put_({ replace_cache_idxs }, gpu_staging_area);

        // 4. Unmask, but now with new nodes!
        cache_mask.index_put_({ new_nids }, true);
    }

    void cacheUpdateFromHolder(torch::Tensor new_nids, torch::Tensor new_feats, torch::Tensor replace_nids, torch::Tensor x)
    {
        ASSERT(use_gpu_transfer, "GPU transfer must be enabled");
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
        cache.index_put_({ replace_cache_idxs }, new_feats);

        // 4. Unmask, but now with new nodes!
        cache_mask.index_put_({ new_nids }, true);
    }

    void worker()
    {
        c10::InferenceMode infer_guard;
        at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, 0);
        at::cuda::CUDAStreamGuard guard(myStream);
        pthread_setname_np(pthread_self(), "CacheManager worker");
        // Starts at 1 just to skip update
        int requests_handled = 1;
        while (worker_alive) {
            bool decay = false;
            bool update = false;

            torch::Tensor nids;
            if(q.wait_and_pop(nids)){

                {
                    // AutoProfiler x("counts");
                    auto counts_acc = counts.accessor<long, 1>();
                    auto nids_acc = nids.accessor<long, 1>();
                    int n = nids.sizes()[0];
                    for(int i = 0; i < n; i++){
                        counts_acc[nids_acc[i]] += 1; 
                    }
                }

                // {
                //     // AutoProfiler x("index put");
                //     counts.index_put_({nids}, counts.index({nids}) + 1);
                // }

                requests_handled++;
                // cout << "handled: " << requests_handled << "\n";
                if(decay_frequency != 0 && requests_handled % decay_frequency == 0){
                    decay = true;
                }                
                if(update_frequency != 0 && requests_handled % update_frequency == 0) {
                    update = true;
                }
            }
            
            if(use_gpu_transfer){
                torch::Tensor new_feats, new_nids;
                if(gpu_feat_holder.tryGet(new_feats, new_nids)){
                    // If we get a new node id, only add if in Top K and not in cache yet
                    auto new_candidate_mask = topk_mask.index({new_nids}) & ~cache_mask.index({new_nids});

                    new_nids = new_nids.masked_select(new_candidate_mask);
                    new_feats = new_feats.index({new_candidate_mask.to(new_feats.device())});


                    auto start2 = high_resolution_clock::now();
                    torch::Tensor replace_cache_idxs = getLeastUsedCacheIndices(new_nids.sizes()[0]);
                    torch::Tensor replace_nids = reverse_mapping.index({replace_cache_idxs});
                    auto stop2 = high_resolution_clock::now();
                    auto duration2 = duration_cast<microseconds>(stop2 - start2);
                    // cout << "size: " << new_nids.sizes()[0] << endl;
                    // cout << "compute replace time: " << duration2.count() << endl;

                    cacheUpdateFromHolder(new_nids, new_feats, replace_nids, replace_cache_idxs);
                }
            }

            // Cache update
            if(update) {
                // cout << "starting stats" << endl;
                // Compute cache statistic

                // auto [values, most_common_nids] = counts.topk(cache_size);
                // auto most_common_mask = torch::zeros(counts.sizes()[0], torch::dtype(torch::kBool));
                // most_common_mask.index_put_({most_common_nids}, true);
                // auto new_candidate_mask = most_common_mask & ~cache_mask;
                // auto replace_nids_mask = ~most_common_mask & cache_mask;
                // torch::Tensor add_nids = big_graph_arange.index({new_candidate_mask});
                // torch::Tensor replace_nids = big_graph_arange.index({replace_nids_mask});

                if(use_gpu_transfer){
                    // Sets topk_mask
                    auto start = high_resolution_clock::now();
                    setTopK();
                    auto stop = high_resolution_clock::now();
                    auto duration = duration_cast<microseconds>(stop - start);
                    cout << "compute topk time: " << duration.count() << endl;
                } else {
                    auto start = high_resolution_clock::now();
                    torch::Tensor add_nids = getMostCommonNodesNotInCache(0);
                    auto stop = high_resolution_clock::now();
                    auto duration = duration_cast<microseconds>(stop - start);
                    cout << "compute topk time: " << duration.count() << endl;
                    add_nids = add_nids.slice(0, 0, staging_area_size);

                    auto start2 = high_resolution_clock::now();
                    torch::Tensor replace_cache_idxs = getLeastUsedCacheIndices(add_nids.sizes()[0]);
                    torch::Tensor replace_nids = reverse_mapping.index({replace_cache_idxs});
                    auto stop2 = high_resolution_clock::now();
                    auto duration2 = duration_cast<microseconds>(stop2 - start2);
                    cout << "compute replace time: " << duration2.count() << endl;
                    // // cout << counts.index({replace_nids.slice(0,0,10)}) << endl;
                    // // torch::Tensor add_nids = getMostCommonNodesNotInCache(staging_area_size);
                    // // torch::Tensor replace_cache_idxs = getLeastUsedCacheIndices(staging_area_size);
                    // // torch::Tensor replace_nids = reverse_mapping.index({replace_cache_idxs});
                    add_nids = add_nids.slice(0, 0, staging_area_size);
                    replace_nids = replace_nids.slice(0, 0, staging_area_size);

                    ASSERT (add_nids.sizes()[0] == replace_nids.sizes()[0], "Internal error, add/replace mismatch " << add_nids.sizes()[0] << " " << replace_nids.sizes()[0] << endl);

                    // Gather and move features to GPU
                    // auto start = high_resolution_clock::now();
                    cacheUpdate(add_nids, replace_nids);
                    // auto stop = high_resolution_clock::now();
                    // auto duration = duration_cast<microseconds>(stop - start);
                    // cout << "update time: " << duration.count() << endl;
                }

            }

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
        }
        std::cout << "worker exited" << std::endl;
        
    }


public:
    CacheManager(const int num_total_nodes, const int cache_size, const int update_frequency, const int decay_frequency, const int staging_area_size, bool use_gpu_transfer)
        : cache_size(cache_size)
        , worker_alive(true)
        , worker_thread(&CacheManager::worker, this)
        , update_frequency(update_frequency)
        , decay_frequency(decay_frequency)
        , staging_area_size(staging_area_size)
        , started_threads(0)
        , finished_threads(0)
        , gpu_feat_holder(staging_area_size)
        , use_gpu_transfer(use_gpu_transfer)
    {
        ASSERT (staging_area_size <= cache_size, "staging_area_size must be smaller than the cache size, staging_area_size: " << staging_area_size << " cache_size: " << cache_size);
        counts = torch::zeros(num_total_nodes, torch::dtype(torch::kLong));
        requests_handled2 = 0;
        topk_mask = torch::zeros(num_total_nodes, torch::dtype(torch::kBool));
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

        cpu_staging_area = torch::empty({staging_area_size, graph_features.sizes()[1]}, torch::device(torch::kCPU).pinned_memory(true).dtype(torch::kFloat32));
        gpu_staging_area = torch::empty(staging_area_size, torch::device(torch::kCUDA).requires_grad(false).dtype(torch::kFloat32));

        big_graph_arange = torch::arange(counts.sizes()[0]);
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
        bool DO_ASYNC = true;
        if(DO_ASYNC){
            q.push(node_ids);    
        } else { }
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

        // std::vector<std::pair<long, long>> v;
        // {
        //     AutoProfiler t("fill pair vec");
        //     auto c_acc = counts.accessor<long, 1>();
        //     v.reserve(counts.sizes()[0]);
        //     for(int i = 0; i < counts.sizes()[0]; i++){
        //         v.push_back(std::make_pair(c_acc[i], i));
        //     }
        // }
        
        // {
        //     AutoProfiler t("nth ele");
        // std::nth_element(v.begin(), v.begin() + cache_size + k, v.end(), std::greater<>{});
        // }

        // std::vector<long> v2;

        // {
        //     AutoProfiler t("fill index vec");
        // for(auto x: v){
        //     v2.push_back(x.second);
        // }
        // }


        // torch::Tensor indices = torch::from_blob(v2.data(), {v2.size()}, torch::dtype(torch::kLong)).clone();


    //    auto [values, indices] = counts.topk(cache_size + k);

    //    // topk_mask has shape indices.shape (cache_size + k), tells us if node in cache or not
    //    torch::Tensor topk_mask = cache_mask.index({indices});
    //    return indices.masked_select(topk_mask.logical_not());

       auto [values, indices] = counts.topk(cache_size + k);
       // topk_mask has shape indices.shape (cache_size + k), tells us if node in cache or not
        torch::Tensor topk_mask = cache_mask.index({indices});
    //    torch::Tensor topk_mask = cache_mask.index({indices});
       return indices.masked_select(topk_mask.logical_not());
    }

    void setTopK(){
        // torch::Tensor ge_mask = counts > 1;
        // cout << "counts size, sum " << counts.sizes() << " " << ge_mask.sum() << endl;

        auto [values, indices] = counts.topk(cache_size);
        topk_mask = torch::zeros(counts.sizes(), torch::dtype(torch::kBool));
        topk_mask.index_put_({indices}, true);
    }

    torch::Tensor getLeastUsedCacheIndices(int k){
        /**
         * Returns least used cache indices based on current counts.
        */

        torch::Tensor counts_in_cache = counts.index({reverse_mapping});

       ASSERT (k <= cache_size, "k must be smaller than the cache size, k: " << k << " cache_size: " << cache_size);
       auto [values, indices] = counts_in_cache.topk(k, 0, false, true);
       return indices;
    }

    torch::Tensor getMostCommonNodes(int k){
        /**
         * Returns most common nids not in cache based on current counts.
        */
        auto [values, indices] = counts.topk(k);
        return indices;
    }

    torch::Tensor getCounts()
    {
        return counts;
    }

    void receiveNewFeatures(torch::Tensor feats, torch::Tensor nids){
        ASSERT(use_gpu_transfer, "GPU transfer must be enabled");
        gpu_feat_holder.setFeats(feats, nids);
    }

    void gilRelease(std::function<void()> f);
};
