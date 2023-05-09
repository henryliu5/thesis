#ifndef SHM_H
#define SHM_H
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <iostream>
#include <vector>

using namespace boost::interprocess;
using std::cout, std::endl;

inline std::string atomic_start_name(int target_store_id, int reader_store_id, int reader_executor_id){
    return "fast_inference_atomic_gpu_" + std::to_string(target_store_id) + "_store_" + std::to_string(reader_store_id) + "_executor_" + std::to_string(reader_executor_id) + "start";
}

inline std::string atomic_finish_name(int target_store_id, int reader_store_id, int reader_executor_id){
    return "fast_inference_atomic_gpu_" + std::to_string(target_store_id) + "_store_" + std::to_string(reader_store_id) + "_executor_" + std::to_string(reader_executor_id) + "finish";
}

inline void shmSetup(int num_stores, int executors_per_store){
    shared_memory_object::remove("fast_inference_shared_mem");
    managed_shared_memory segment(create_only, "fast_inference_shared_mem", 65536);

    for(int i = 0; i < num_stores; i++){
        auto lock_name = ("fast_inference_mutex_gpu_" + std::to_string(i)).c_str();
        cout << "Constructing lock: " << lock_name << endl;
        segment.construct<interprocess_sharable_mutex>(lock_name)();

        for(int j = 0; j < num_stores; j++){
            for(int k = 0; k < executors_per_store; k++){
                auto start_name = atomic_start_name(i, j, k).c_str();
                segment.construct<std::atomic<int>>(start_name)(0);

                auto finish_name = atomic_finish_name(i, j, k).c_str();
                segment.construct<std::atomic<int>>(finish_name)(0);
            }
        }
    }
}

class SHMLocks{

    std::vector<interprocess_sharable_mutex*> interprocess_mutexes;
    managed_shared_memory segment;

public:

    SHMLocks() : segment(open_only, "fast_inference_shared_mem")
        {
        const int MAX_STORES = 64;
        for(int i = 0; i < MAX_STORES; i++){
            auto lock_name = ("fast_inference_mutex_gpu_" + std::to_string(i)).c_str();
            interprocess_sharable_mutex* mutex = segment.find<interprocess_sharable_mutex>(lock_name).first;
            if(mutex == nullptr){
                break;
            }
            cout << "Finding lock: " << lock_name << endl;
            interprocess_mutexes.push_back(mutex);
        }
    }

    void readLock(int index){
        interprocess_mutexes[index]->lock_sharable();
    }

    void readUnlock(int index){
        interprocess_mutexes[index]->unlock_sharable();
    }

    void writeLock(int index){
        interprocess_mutexes[index]->lock();
    }

    void writeUnlock(int index){
        interprocess_mutexes[index]->unlock();
    }
};

#endif