export PYTHONPATH=$PYTHONPATH:/home/henry/thesis
export PYTHONPATH=$PYTHONPATH:/home/henry/thesis/build
export TORCH_USE_RTLD_GLOBAL=YES
sudo nvidia-smi -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d

for i in 2 4 6 8
do
for j in {2..2}
do
# python benchmark/multi_process.py -c static -t 5 -e $i -g $j -o throughput_testing 
# sleep 1
# python benchmark/multi_process.py -c count -t 5 -e $i -g $j -o throughput_testing
# sleep 1
python benchmark/multi_process.py -c cpp -t 10 -e $i -g $j -o throughput_testing
sleep 1
python benchmark/multi_process.py -c cpp_lock -t 10 -e $i -g $j -o throughput_testing
sleep 1
# python benchmark/multi_process.py -c static -t 10 -e $i -g $j -o throughput_testing -b 0.8
# sleep 1
# python benchmark/multi_process.py -c count -t 10 -e $i -g $j -o throughput_testing -b 0.8
# sleep 1
# python benchmark/multi_process.py -c cpp -t 10 -e $i -g $j -o throughput_testing -b 0.8
# sleep 1
# python benchmark/multi_process.py -c cpp_lock -t 10 -e $i -g $j -o throughput_testing -b 0.8
# sleep 1
done
done

sudo echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -c DEFAULT
exit