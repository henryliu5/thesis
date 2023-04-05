#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstddef>
#include <cassert>
#include <utility>
#include <iostream>

#include <chrono>
#include <atomic>
#include <thread>
using namespace std::chrono;

int main(int argc, char *argv[])
{
   using namespace boost::interprocess;
   typedef std::pair<double, int> MyType;
    int n = 1000000;
    
   if(argc == 1){  //Parent process
      //Remove shared memory on construction and destruction
      struct shm_remove
      {
         shm_remove() { shared_memory_object::remove("MySharedMemory"); }
         ~shm_remove(){ shared_memory_object::remove("MySharedMemory"); }
      } remover;

      //Construct managed shared memory
      managed_shared_memory segment(create_only, "MySharedMemory", 65536);

    volatile bool* parent_done = segment.construct<bool>("parent done")(false);
    volatile bool* child_start = segment.construct<bool>("child start")(false);
    volatile bool* child_done = segment.construct<bool>("child done")(false);
    
      volatile std::atomic<long> *atomic_long_int = segment.construct<std::atomic<long>>
         ("atomic long")  //name of the object
         (0);            //ctor first argument

      volatile long *long_int = segment.construct<long>
         ("not atomic long")  //name of the object
         (0);            //ctor first argument


    while(!*child_start);

      for(int i = 0; i < n; i++){
        atomic_long_int->fetch_add(1);
        *long_int += 1;
      }

    *parent_done = true;
    while(!*child_done);

      std::cout << "atomic " << atomic_long_int->load() << std::endl;

      std::cout << "parent atomic value: " << (long) segment.find<std::atomic<long>>("atomic long").first->load() << std::endl;
      std::cout << "parent non-atomic value: " << (long) *segment.find<std::atomic<long>>("not atomic long").first << std::endl;

    //   //Check child has destroyed all objects
    //   if(segment.find<MyType>("MyType array").first ||
    //      segment.find<MyType>("MyType instance").first ||
    //      segment.find<MyType>("MyType array from it").first)
    //      return 1;
   }
   else{
      //Open managed shared memory
      managed_shared_memory segment(open_only, "MySharedMemory");

      std::pair<MyType*, managed_shared_memory::size_type> res;

      auto atom = segment.find<std::atomic<long>>("atomic long");
      auto not_atom = segment.find<long>("not atomic long");
      volatile bool* parent_done = segment.find<bool>("parent done").first;
      volatile bool* child_done = segment.find<bool>("child done").first;
      volatile bool* child_start = segment.find<bool>("child start").first;

        volatile std::atomic<long>* atomic_long_int = atom.first;
        volatile long* long_int = not_atom.first;

      *child_start = true;



      for(int i = 0; i < n; i++){
        atomic_long_int->fetch_add(1);
        *long_int += 1;
      }

    *child_done = true;

    while(!*parent_done);

      std::cout << "atomic " << atomic_long_int->load() << std::endl;

      std::cout << "child atomic value: " << (long) segment.find<std::atomic<long>>("atomic long").first->load() << std::endl;
      std::cout << "child non-atomic value: " << (long) *segment.find<std::atomic<long>>("not atomic long").first << std::endl;
   }
   return 0;
}