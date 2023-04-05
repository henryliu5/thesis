export PYTHONPATH=$PYTHONPATH:/home/henry/thesis
export PYTHONPATH=$PYTHONPATH:/home/henry/thesis/build
export TORCH_USE_RTLD_GLOBAL=YES
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sudo nvidia-smi -i 1 -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d

for i in {1..7}
do
for j in {2..2}
do
python benchmark/multi_process.py -c static -t 10 -e $i -g $j -o throughput_testing
sleep 5
python benchmark/multi_process.py -c count -t 10 -e $i -g $j -o throughput_testing
sleep 5
python benchmark/multi_process.py -c cpp -t 10 -e $i -g $j -o throughput_testing
sleep 5
python benchmark/multi_process.py -c cpp_lock -t 10 -e $i -g $j -o throughput_testing
sleep 5
done
done

sudo echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT
sudo nvidia-smi -i 1 -c DEFAULT
exit