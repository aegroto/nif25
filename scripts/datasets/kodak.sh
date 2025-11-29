#!/bin/bash
> schedule.sh

for config in $(find configurations/.nif/kodak/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"

    for i in {1..24}
    do
        log_file="logs/${config_id}_$i.txt"
        echo "./scripts/run/full_experiment.sh \"$config\" test_images/kodak/$i.png \"results/nif/kodak/$config_id/$i\" > "logs/${config_id}_${i}_full.txt" 2>&1" >> schedule.sh
    done
done
