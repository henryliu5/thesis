export PYTHONPATH=$PYTHONPATH:/home/henry/thesis
export PYTHONPATH=$PYTHONPATH:/home/henry/thesis/build
export TORCH_USE_RTLD_GLOBAL=YES
sudo nvidia-smi -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d


# for i in 8
# do
# for j in {1..2}
# do
# python benchmark/multi_process.py -c static -t 5 -e $i -g $j -o throughput_direct  -d
# sleep 1
# python benchmark/multi_process.py -c count -t 5 -e $i -g $j -o throughput_direct -d
# sleep 1
# python benchmark/multi_process.py -c cpp -t 1 -e $i -g $j -o throughput_direct -d
# sleep 1
# exit
# python benchmark/multi_process.py -c cpp_lock -t 2 -e $i -g $j -o throughput_direct -d
# sleep 1
# exit
# python benchmark/multi_process.py -c static -t 10 -e $i -g $j -o throughput_direct -b 0.8 -d
# sleep 1
# python benchmark/multi_process.py -c count -t 10 -e $i -g $j -o throughput_direct -b 0.8 -d
# sleep 1
# python benchmark/multi_process.py -c cpp -t 10 -e $i -g $j -o throughput_direct -b 0.8 -d
# sleep 1
# python benchmark/multi_process.py -c cpp_lock -t 10 -e $i -g $j -o throughput_direct -b 0.8 -d
# sleep 1
# done
# done


for i in 1 2 4 8
do
for j in 1 2
do
python benchmark/multi_process.py -c static -t 15 -e $i -g $j -o multiple_throughput 
sleep 1
python benchmark/multi_process.py -c count -t 15 -e $i -g $j -o multiple_throughput
sleep 1
python benchmark/multi_process.py -c cpp -t 15 -e $i -g $j -o multiple_throughput
sleep 1
python benchmark/multi_process.py -c cpp_lock -t 15 -e $i -g $j -o multiple_throughput
sleep 1
python benchmark/multi_process.py -c static -t 15 -e $i -g $j -o multiple_throughput -b 0.8
sleep 1
python benchmark/multi_process.py -c count -t 15 -e $i -g $j -o multiple_throughput -b 0.8
sleep 1
python benchmark/multi_process.py -c cpp -t 15 -e $i -g $j -o multiple_throughput -b 0.8
sleep 1
python benchmark/multi_process.py -c cpp_lock -t 15 -e $i -g $j -o multiple_throughput -b 0.8
sleep 1
done
done

sudo echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -c DEFAULT
exit