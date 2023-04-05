export PYTHONPATH=$PYTHONPATH:/home/henry/thesis
export PYTHONPATH=$PYTHONPATH:/home/henry/thesis/build
export TORCH_USE_RTLD_GLOBAL=YES
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sudo nvidia-smi -i 1 -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d

# for i in {1..100}
# do
#     # python benchmark/infer_cache.py -c static -p 0.2 -t 3 -o testing/trial_$i --use_pinned_mem
#     # python benchmark/infer_cache.py -c count -p 0.2 -t 3 -o testing/trial_$i --use_pinned_mem
#     # python benchmark/infer_cache.py -c cpp -p 0.2 -t 3 -o testing/trial_$i --use_pinned_mem
#     python benchmark/multi_process.py -c static -o multi_testing/trial_$i
#     python benchmark/multi_process.py -c count -o multi_testing/trial_$i
#     python benchmark/multi_process.py -c cpp -o multi_testing/trial_$i
# done
# exit


for i in {5..8}
do
for j in {1..2}
do
python benchmark/multi_process.py -c static -t 10 -e $i -g $j -o throughput_testing
python benchmark/multi_process.py -c count -t 10 -e $i -g $j -o throughput_testing
python benchmark/multi_process.py -c cpp -t 10 -e $i -g $j -o throughput_testing
python benchmark/multi_process.py -c cpp_lock -t 10 -e $i -g $j -o throughput_testing
done
done
# python benchmark/multi_process.py -c static -b 0.8 -t 5
# python benchmark/multi_process.py -c count -b 0.8 -t 5
# python benchmark/multi_process.py -c cpp -b 0.8  -t 5
# python benchmark/multi_process.py -c cpp_lock -b 0.8 -t 5

sudo echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT
sudo nvidia-smi -i 1 -c DEFAULT
exit

python benchmark/infer_cache.py -c static -b 0.8 -p 0.1 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c count -b 0.8 -p 0.1 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c cpp -b 0.8 -p 0.1 -t 3 -o testing --use_pinned_mem

python benchmark/infer_cache.py -c static -p 0.1 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c count -p 0.1 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c cpp -p 0.1 -t 3 -o testing --use_pinned_mem

python benchmark/infer_cache.py -c static -b 0.8 -p 0.2 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c count -b 0.8 -p 0.2 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c cpp -b 0.8 -p 0.2 -t 3 -o testing --use_pinned_mem

python benchmark/infer_cache.py -c static -p 0.2 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c count -p 0.2 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c cpp -p 0.2 -t 3 -o testing --use_pinned_mem

python benchmark/infer_cache.py -c lfu -b 0.8 -p 0.1 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c lfu -p 0.1 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c lfu -b 0.8 -p 0.2 -t 3 -o testing --use_pinned_mem
python benchmark/infer_cache.py -c lfu -p 0.2 -t 3 -o testing --use_pinned_mem


while true; do
python benchmark/infer_cache.py -c baseline -p 0.2 -o fast_sampling -t 10
python benchmark/infer_cache.py -c static -p 0.2 -o fast_sampling -t 10
python benchmark/infer_cache.py -c cpp -p 0.2 -o fast_sampling -t 10
# python benchmark/infer_cache.py -c baseline -b 0.8 -p 0.2 -o fast_sampling --profile
# python benchmark/infer_cache.py -c baseline -p 0.2 -o fast_sampling --use_pinned_mem --profile
python benchmark/infer_cache.py -c static -p 0.2 -o fast_sampling --use_pinned_mem --profile
# python benchmark/infer_cache.py -c baseline -b 0.8 -p 0.2 -o fast_sampling --use_pinned_mem --profile
# python benchmark/infer_cache.py -c cpp -p 0.2 -o fast_sampling --use_pinned_mem --profile
done

# # # no pinned memory tests
# python benchmark/infer_cache.py -c static
# python benchmark/infer_cache.py -c count
# python benchmark/infer_cache.py -c lfu
# python benchmark/infer_cache.py -c async

# python benchmark/infer_cache.py -c static -b 0.8
# python benchmark/infer_cache.py -c count -b 0.8
# python benchmark/infer_cache.py -c lfu -b 0.8
# python benchmark/infer_cache.py -c async -b 0.8

# # tests with pinned memory
# python benchmark/infer_cache.py -c static --use_pinned_mem
# python benchmark/infer_cache.py -c count --use_pinned_mem
# python benchmark/infer_cache.py -c lfu --use_pinned_mem
# python benchmark/infer_cache.py -c async --use_pinned_mem

# python benchmark/infer_cache.py -c static -b 0.8 --use_pinned_mem
# python benchmark/infer_cache.py -c count -b 0.8 --use_pinned_mem
# python benchmark/infer_cache.py -c lfu -b 0.8 --use_pinned_mem
# python benchmark/infer_cache.py -c async -b 0.8 --use_pinned_mem

# # repeat all but with profiling
# # no pinned memory tests
# # python benchmark/infer_cache.py -c static --profile
# # python benchmark/infer_cache.py -c count --profile
# # python benchmark/infer_cache.py -c lfu --profile
# # python benchmark/infer_cache.py -c async --profile

# # python benchmark/infer_cache.py -c static -b 0.8 --profile
# # python benchmark/infer_cache.py -c count -b 0.8 --profile
# # python benchmark/infer_cache.py -c lfu -b 0.8 --profile
# # python benchmark/infer_cache.py -c async -b 0.8 --profile

# # # tests with pinned memory
# # python benchmark/infer_cache.py -c static --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c count --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c lfu --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c async --use_pinned_mem --profile

# # python benchmark/infer_cache.py -c static -b 0.8 --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c count -b 0.8 --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c lfu -b 0.8 --use_pinned_mem --profile
# # python benchmark/infer_cache.py -c async -b 0.8 --use_pinned_mem --profile