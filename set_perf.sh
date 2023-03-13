# Disable turbo boost
sudo echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo

# Set all CPU's to performance mode
for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do
  sudo echo performance > $i
done

# Disable ASLR
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space