# Disable turbo boost
sudo echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo

# Set all CPU's to performance mode
for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do
  sudo echo performance > $i
done

# Disable SMT
# echo off | sudo tee /sys/devices/system/cpu/smt/control

# Disable ASLR
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# Set enough mmap 
sudo sysctl -w vm.max_map_count=640000