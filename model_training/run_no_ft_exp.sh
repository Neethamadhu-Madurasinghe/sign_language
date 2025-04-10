#!/bin/bash

# List of instance values
instance_values=(6 4 3 2)

# Number of runs
n_runs=1

# Loop over each instance value
for instances in "${instance_values[@]}"; do
    echo "Running experiments for instances=$instances"
    
    for ((runid=0; runid<n_runs; runid++)); do
        echo "  --> python no_finetunning.py --runid $runid --instances $instances"
        python no_finetunning.py --runid "$runid" --instances "$instances"
    done
done
