#ifndef CACHE_MANAGER_H
#define CACHE_MANAGER_H
// #include <boost/lockfree/spsc_queue.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/named_sharable_mutex.hpp>
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
#include "shm.hpp"
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
    int BOUND = 7;
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

using namespace boost::interprocess;

class CacheManager {
    /** Controls cache state.
     *  Computes cache usage statistics and performs dynamic cache updates.
     */
private:
    int cache_size;
    int num_engines;
    int device_id;
    // int update_frequency;
    // int decay_frequency;
    // int staging_area_size;

    // TODO switch to Boost lockfree queue if multiple producers
    // boost::lockfree::spsc_queue<torch::Tensor, boost::lockfree::capacity<1024>> q;
    // moodycamel::ConcurrentQueue<torch::Tensor> q;
    // concurrent_queue<torch::Tensor> q;

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

    std::vector<interprocess_sharable_mutex*> interprocess_mutexes;
    interprocess_sharable_mutex* local_ipc_mutex;
    managed_shared_memory segment;
    bool use_locking;

    std::vector<std::vector<std::atomic<int>*>> start_atomics;
    std::vector<std::vector<std::atomic<int>*>> finish_atomics;
    int total_stores;
    int executors_per_store;

    std::vector<std::vector<std::vector<std::atomic<int>*>>> all_start_atomics;
    std::vector<std::vector<std::vector<std::atomic<int>*>>> all_finish_atomics;

    // CacheManager specific cache metadata
    torch::Tensor counts;
    std::atomic<long> started_threads;
    std::atomic<long> finished_threads;

    // Test variables
    bool use_gpu_transfer;
    torch::Tensor topk_mask;
    concurrent_queue<std::tuple<torch::Tensor, torch::Tensor>> gpu_q;
    torch::Tensor cache_candidate_mask;
    std::mutex cache_mutex;


public:
    CacheManager(const int num_total_nodes, const int cache_size, const int device_id, const int num_engines, bool use_gpu_transfer, bool use_locking, const int total_stores, const int executors_per_store)
        : cache_size(cache_size)
        , worker_alive(true)
        , gpu_q()
        , worker_thread{}
        // , update_frequency(update_frequency)
        // , decay_frequency(decay_frequency)
        // , staging_area_size(staging_area_size)
        , started_threads(0)
        , finished_threads(0)
        // , gpu_feat_holder(staging_area_size)
        , use_gpu_transfer(use_gpu_transfer)
        , num_engines(num_engines)
        , device_id(device_id)
        , segment(open_only, "fast_inference_shared_mem")
        , local_ipc_mutex(0)
        , use_locking(use_locking)
        , total_stores(total_stores)
        , executors_per_store(executors_per_store)
    {
        auto lock_name = ("fast_inference_mutex_gpu_" + std::to_string(device_id));

        local_ipc_mutex = segment.find<interprocess_sharable_mutex>(lock_name.c_str()).first;
        if(local_ipc_mutex == 0){
            cout << "Failed to find lock in shm, dev id: " << device_id << " couldn't find " << lock_name << endl;
            exit(1);
        }

        // ASSERT (staging_area_size <= cache_size, "staging_area_size must be smaller than the cache size, staging_area_size: " << staging_area_size << " cache_size: " << cache_size);
        counts = torch::zeros(num_total_nodes, torch::dtype(torch::kLong));
        // requests_handled2 = 0;
        topk_mask = torch::zeros(num_total_nodes, torch::dtype(torch::kBool));

        cout << "Setting local mutex to " << "fast_inference_mutex_gpu_" + std::to_string(device_id) << endl;
        for(int i = 0; i < num_engines; i++){
            interprocess_mutexes.push_back(segment.find<interprocess_sharable_mutex>(("fast_inference_mutex_gpu_" + std::to_string(i)).c_str()).first);
        }

        for(int target = 0; target < total_stores; target++){
            std::vector<std::vector<std::atomic<int>*>> target_start;
            std::vector<std::vector<std::atomic<int>*>> target_finish;
            for(int i = 0; i < total_stores; i++){
                std::vector<std::atomic<int>*> start_v;
                std::vector<std::atomic<int>*> finish_v;
                for(int j = 0; j < executors_per_store; j++){
                    auto start_name = atomic_start_name(target, i, j).c_str();
                    start_v.push_back(segment.find<std::atomic<int>>(start_name).first);

                    auto finish_name = atomic_finish_name(target, i, j).c_str();
                    finish_v.push_back(segment.find<std::atomic<int>>(finish_name).first);
                }

                target_start.push_back(start_v);
                target_finish.push_back(finish_v);

                if(target == device_id){
                    // This store's atomics
                    start_atomics.push_back(start_v);
                    finish_atomics.push_back(finish_v);
                }
                
            }
            all_start_atomics.push_back(target_start);
            all_finish_atomics.push_back(target_finish);
        }


        //!! Start thread after so construction is done
        worker_thread = std::thread(&CacheManager::smallWorker, this);
    }

    ~CacheManager()
    {
        gilRelease([this](){
            std::cout << "entered destructor" << std::endl;
            // while(!q.empty()); // This waits for worker to finish processing
            worker_alive = false;
            // q.disable();
            gpu_q.disable();

            std::cout << "bool set, calling join" << std::endl;
            worker_thread.join();
        }
        );
        std::cout << "exiting destructor" << std::endl;
    }

    void setCache(torch::Tensor graph_features, torch::Tensor cache_mask, torch::Tensor cache_mapping,
        torch::Tensor reverse_mapping, torch::Tensor cache)
    {
        this->graph_features = graph_features;
        this->cache_mask = cache_mask;
        this->cache_mapping = cache_mapping;
        this->reverse_mapping = reverse_mapping;
        this->cache = cache;

        // cpu_staging_area = torch::empty({staging_area_size, graph_features.sizes()[1]}, torch::device(torch::kCPU).pinned_memory(true).dtype(torch::kFloat32));
        // gpu_staging_area = torch::empty(staging_area_size, torch::device(torch::kCUDA).requires_grad(false).dtype(torch::kFloat32));

        // big_graph_arange = torch::arange(counts.sizes()[0], torch::device(torch::kCUDA));
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
    void threadEnter(int target_id, int reader_store_id, int reader_executor_id)
    {
        // started_threads++;
        // auto atomic_name = atomic_start_name(target_id, reader_store_id, reader_executor_id).c_str();
        // std::atomic<int>* a = segment.find<std::atomic<int>>(atomic_name).first;
        // auto x = a->fetch_add(1);
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        all_start_atomics[target_id][reader_store_id][reader_executor_id]->fetch_add(1);
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "start incr = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        // cout << "started " << target_id << " " << reader_store_id << " " << reader_executor_id << " " << x + 1 << endl;
    }

    void threadExit(int target_id, int reader_store_id, int reader_executor_id)
    {
        // finished_threads++;
        // cout << "finished: " << finished_threads << "\n";
        // auto atomic_name = atomic_finish_name(target_id, reader_store_id, reader_executor_id).c_str();
        // std::atomic<int>* a = segment.find<std::atomic<int>>(atomic_name).first;
        // auto x = a->fetch_add(1);
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        all_finish_atomics[target_id][reader_store_id][reader_executor_id]->fetch_add(1);
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "finish incr = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        // cout << "finished " << target_id << " " << reader_store_id << " " << reader_executor_id << " " << x + 1 << endl;
    }

    void placeFeatsInQueue(torch::Tensor feats, torch::Tensor nids){
        ASSERT(use_gpu_transfer, "GPU transfer must be enabled");
        gpu_q.push({feats, nids});
    }

    void setCacheCandidates(torch::Tensor c){
        this->cache_candidate_mask = c;
    }

    void readLock(int index){
        ASSERT(index >= 0 && index < interprocess_mutexes.size(), "Acquiring lock out of range");
        interprocess_mutexes[index]->lock_sharable();
    }

    void readUnlock(int index){
        interprocess_mutexes[index]->unlock_sharable();
    }

    void writeLock(){
        local_ipc_mutex->lock();
    }

    void writeUnlock(){
        local_ipc_mutex->unlock();
    }

    void smallWorker()
        {
            try{
            c10::InferenceMode infer_guard;
            at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, device_id);
            at::cuda::CUDAStreamGuard guard(myStream);
            pthread_setname_np(pthread_self(), "CacheManager smallWorker");

            // auto cache_size_buf = torch::empty({cache_size}, torch::device(torch::kCPU).pinned_memory(true).dtype(torch::kLong));

            while (worker_alive) {
                #ifndef NDEBUG
                cout << "WARNING: cache update compiled in DEBUG mode: " << syscall(__NR_gettid) << endl;
                #endif

                std::tuple<torch::Tensor, torch::Tensor> p;
                if(gpu_q.wait_and_pop(p)){
                    // TODO add setting to enable mutex or use atomics
                    // cache_mutex.lock();
                    if(use_locking){
                        myStream.synchronize();
                        ASSERT(local_ipc_mutex != 0, "failed pointer nonzero");
                        local_ipc_mutex->lock();
                    } else {
                        // local_ipc_mutex->lock();
                        if(!local_ipc_mutex->try_lock()){
                            // Skip this update if failed to get lock
                            continue;
                        }
                    }
                    /**
                     * Needed:
                     * - is_cache_candidate - missing
                     * - cache_mask - done
                     * - big_graph_arange - done
                     * - cache_mapping - done
                     * - most_common_nids?
                    */
                    torch::Tensor new_feats = std::get<0>(p);
                    torch::Tensor new_nids = std::get<1>(p);

                    // Not actually pinned, won't be async
                    // new_nids = new_nids.to(torch::device(torch::kCUDA), true);

                    auto cache_mask_device = cache_mask;//.to(torch::device(torch::kCUDA), true);
                    // auto cache_mask_device = cache_mapping >= 0;
                    ASSERT(std::get<0>(at::_unique(new_nids)).sizes() == new_nids.sizes(), "new nids must be unique");
                    ASSERT(new_nids.dtype() == torch::kLong, "new nids must be longs");
                    ASSERT(new_nids.max().item<long>() < cache_candidate_mask.sizes()[0], "Out of bounds, max: " << new_nids.max().item<long>() << " length " << cache_candidate_mask.sizes()[0]);
                    ASSERT(new_nids.min().item<long>() >= 0 && new_nids.max().item<long>() < cache_candidate_mask.sizes()[0], "new_nids out of bounds, min " << new_nids.min().item<long>() << " max, " << new_nids.max().item<long>() << " indexing into " << cache_candidate_mask.sizes()[0]);

                    auto new_nid_mask = cache_candidate_mask.index({new_nids});

                    //!! This is necessary if operations are super racy... it is possible for the node ids given to be stale
                    //!! maybe just want to throw this update away if this is the case
                    new_nid_mask &= ~cache_mask_device.index({new_nids});

                    // TODO figure out "usefulness" threshold, can leave right after computing nids to add shape
                    auto nids_to_add = new_nids.index({new_nid_mask});

                    const float THRESHOLD = 0.01;
                    if((float) nids_to_add.sizes()[0] / (float) cache_size <= THRESHOLD){
                        if(use_locking){
                            myStream.synchronize();
                            local_ipc_mutex->unlock();
                        } else {
                            local_ipc_mutex->unlock();
                        }
                        continue;
                    }

                    new_feats = new_feats.index({new_nid_mask});

                    ASSERT((cache_mapping.index({nids_to_add}) < 0).all(0).item<bool>(), "Trying to add node to cache already present");
                    ASSERT((cache_mapping.index({reverse_mapping}) >= 0).all(0).item<bool>(), "Reverse mapping invalid");
                    auto replace_nid_mask = ~(cache_candidate_mask.index({reverse_mapping}));
                    auto replace_nids = reverse_mapping.index({replace_nid_mask});
                    ASSERT((cache_mapping.index({replace_nids}) >= 0).all(0).item<bool>(), "Replace nids invalid");

                    auto num_to_add = std::min(replace_nids.sizes()[0], std::min(nids_to_add.sizes()[0], (const long) cache_size));

                    // TODO figure out "usefulness" threshold
                    if(num_to_add == 0){
                        // cache_mutex.unlock();
                        if(use_locking){
                            myStream.synchronize();
                            local_ipc_mutex->unlock();
                        } else {
                            local_ipc_mutex->unlock();
                        }
                        continue;
                    }

                    replace_nids = replace_nids.slice(0, 0, num_to_add);
                    nids_to_add = nids_to_add.slice(0, 0, num_to_add);

                    ASSERT(replace_nids.min().item<long>() >= 0 && replace_nids.max().item<long>() < cache_mask_device.sizes()[0], "replace_nids out of bounds");
                    // Blind write 0's into cache mask
                    cache_mask_device.index_put_({replace_nids}, false);
                    myStream.synchronize();

                    // 2. Wait for enough threads to finish
                    std::vector<std::vector<int>> atomics_at_start;
                    for(int i = 0; i < total_stores; i++){
                        std::vector<int> v;
                        for(int j = 0; j < executors_per_store; j++){
                            auto atomic_name = atomic_start_name(device_id, i, j).c_str();
                            std::atomic<int>* a = segment.find<std::atomic<int>>(atomic_name).first;
                            v.push_back(a->load());
                        }
                        atomics_at_start.push_back(v);
                    }

                    // 3. Spin on atomics
                    for(int i = 0; i < total_stores; i++){
                        for(int j = 0; j < executors_per_store; j++){
                            auto atomic_name = atomic_finish_name(device_id, i, j).c_str();
                            std::atomic<int>* a = segment.find<std::atomic<int>>(atomic_name).first;
                            //!! Need to make sure worker is alive in this loop!
                            while (a->load() < atomics_at_start[i][j] && worker_alive){
                                std::this_thread::sleep_for(std::chrono::microseconds(100));
                            }
                        }
                    }

                    auto cache_slots = cache_mapping.index({replace_nids});
                    ASSERT(cache_slots.min().item<long>() >= 0 && cache_slots.max().item<long>() < cache_size, "cache slots out of bounds, min " << cache_slots.min().item<long>() << " max " << cache_slots.max().item<long>());

                    cache_mapping.index_put_({replace_nids}, -1);
                    cache_mapping.index_put_({nids_to_add}, cache_slots);
                    reverse_mapping.index_put_({ cache_slots }, nids_to_add);
                    ASSERT((cache_mapping.index({nids_to_add}) >= 0).all(0).item<bool>(), "Reverse mapping invalid");
                    ASSERT((cache_mapping.index({reverse_mapping}) >= 0).all(0).item<bool>(), "Reverse mapping update invalid rev: " << reverse_mapping << " cache map[rev mapping] " << cache_mapping.index({reverse_mapping}));

                    cache.index_put_({cache_slots}, new_feats.slice(0, 0, num_to_add));

                    ASSERT(nids_to_add.min().item<long>() >= 0 && nids_to_add.max().item<long>() < cache_mask_device.sizes()[0], "nids to add out of bounds");
                    // Now write 1's
                    cache_mask_device.index_put_({nids_to_add}, true);

                    // cache_mutex.unlock();
                    if(use_locking){
                        myStream.synchronize();
                        local_ipc_mutex->unlock();
                    } else {
                        myStream.synchronize();
                        local_ipc_mutex->unlock();
                    }
                }
                
            }
            }
            catch (const c10::Error& e) {   cout << e.what() << endl; throw std::invalid_argument("Failed to load model: " + e.msg()); }
            catch (const boost::exception& e)
            {
                std::string diag = diagnostic_information(e);
                // display your error message here, then do whatever you need to, e.g.        
                cout << "boost exception" << diag << endl;
                exit(1);
            }
            std::cout << "worker exited" << std::endl;
        }

    void gilRelease(std::function<void()> f);
};

#endif