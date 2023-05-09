export PYTHONPATH=$PYTHONPATH:/home/henry/thesis
export PYTHONPATH=$PYTHONPATH:/home/henry/thesis/build
export TORCH_USE_RTLD_GLOBAL=YES
sudo nvidia-smi -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d

# for i in {50..350..50}
# do
# python benchmark/multi_process.py -c static -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i -b 0.8
# sleep 1
# python benchmark/multi_process.py -c count -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i -b 0.8
# sleep 1
# python benchmark/multi_process.py -c cpp -t 10 -e 8 -g 2 -o p99_latency/rate_$i -r $i -b 0.8
# sleep 1
# python benchmark/multi_process.py -c cpp_lock -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i -b 0.8
# sleep 1
# done

# for i in 400 500 600 700 800 900 1000
# do
# python benchmark/multi_process.py -c static -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i -b 0.8
# sleep 1
# python benchmark/multi_process.py -c count -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i -b 0.8
# sleep 1
# python benchmark/multi_process.py -c cpp -t 10 -e 8 -g 2 -o p99_latency/rate_$i -r $i -b 0.8
# sleep 1
# python benchmark/multi_process.py -c cpp_lock -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i -b 0.8
# sleep 1
# done

for i in 700 800 900 1000
do
python benchmark/multi_process.py -c static -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i
sleep 1
python benchmark/multi_process.py -c count -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i
sleep 1
python benchmark/multi_process.py -c cpp -t 10 -e 8 -g 2 -o p99_latency/rate_$i -r $i
sleep 1
python benchmark/multi_process.py -c cpp_lock -t 5 -e 8 -g 2 -o p99_latency/rate_$i -r $i
sleep 1
done

sudo echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -c DEFAULT
exit