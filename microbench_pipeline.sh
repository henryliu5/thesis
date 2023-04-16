export PYTHONPATH=$PYTHONPATH:/home/henry/thesis
export PYTHONPATH=$PYTHONPATH:/home/henry/thesis/build
export TORCH_USE_RTLD_GLOBAL=YES
sudo nvidia-smi -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d

for i in 1 2 4 8
do
for j in 1
do
python benchmark/pipeline_microbench.py -c cpp_lock -s $j -e $i
done
done

sudo echo quit | nvidia-cuda-mps-control
sudo nvidia-smi -c DEFAULT
exit