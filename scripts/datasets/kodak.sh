#!/bin/bash
> schedule.sh

# for config in $(find configurations/.base/*.yaml)
# for config in $(find configurations/.tuning/*.yaml)
# for config in "configurations/default.yaml"
for config_id in "176_32_4_0" "320_64_4.5_6"
do
    for loss in "l1_ssim" "logcosh" "visual"
    do
        # config_file=$(basename $config)
        # config_id="${config_file%.yaml}"
        config="configurations/.nif/kodak/$config_id.yaml"

        # echo "cp -rL results/nif/kodak/default results/nif/kodak/${config_id}" >> schedule.sh
        # base_config_id=${config_id::-3}
        # echo "cp -rL results/nif/kodak_base/${base_config_id} results/nif/kodak/${config_id}" >> schedule.sh

        for i in {1..24}
        # for i in 24 1 7 8 20
        do
            log_file="logs/${config_id}_$i.txt"
            # echo "./scripts/run/fit_experiment.sh \"$config\" test_images/kodak/$i.png \"results/nif/kodak/$config_id/$i\" > "logs/${config_id}_${i}_fitting.txt" 2>&1" >> schedule.sh
            # echo "./scripts/run/quantization_experiment.sh \"$config\" test_images/kodak/$i.png \"results/nif/kodak/$config_id/$i\" > "logs/${config_id}_${i}_quantization.txt" 2>&1" >> schedule.sh
            # echo "./scripts/run/compression_experiment.sh \"$config\" test_images/kodak/$i.png \"results/nif/kodak/$config_id/$i\" > "logs/${config_id}_${i}_compression.txt" 2>&1" >> schedule.sh

            # echo "./scripts/run/full_experiment.sh \"$config\" test_images/kodak/$i.png \"results/nif/kodak/$config_id/$i\" > "logs/${config_id}_${i}_full.txt" 2>&1" >> schedule.sh
            echo "./scripts/run/decode_only_experiment.sh \"$config\" test_images/kodak/$i.png \"results/nif/kodak/$loss/${loss}_$config_id/$i\" > "logs/${loss}_${config_id}_${i}_decode_only.txt" 2>&1" >> schedule.sh
        done
    done

    for config in $(find configurations/.tuning/*.yaml)
    do
        break

        config_file=$(basename $config)
        config_id="${config_file%.yaml}"

        # base_config_id=${config_id::20}
        # echo "cp -rL results/nif/kodak/${base_config_id} results/nif/kodak/${config_id}" >> schedule.sh

        for i in {1..24}
        # for i in 24 1 7 8 20
        do
            log_file="logs/${config_id}_$i.txt"
            # echo "mkdir -p $PWD/results/nif/kodak/${config_id}/$i/" >> schedule.sh
            # echo "ln -s $PWD/results/nif/kodak/default/$i/fitted/ $PWD/results/nif/kodak/${config_id}/$i/" >> schedule.sh
            echo "./scripts/run/quantization_experiment.sh \"$config\" test_images/kodak/$i.png \"results/nif/kodak/$config_id/$i\" > "logs/${config_id}_${i}_quantization.txt" 2>&1" >> schedule.sh
            echo "./scripts/run/compression_experiment.sh \"$config\" test_images/kodak/$i.png \"results/nif/kodak/$config_id/$i\" > "logs/${config_id}_${i}_compression.txt" 2>&1" >> schedule.sh
        done
    done
done

# echo "python calculate_summary.py compressed/stats.json results/summaries/kodak/tmp/compressed.json results/nif/kodak" >> schedule.sh
# echo "python calculate_summary.py fitted/stats.json results/summaries/kodak/tmp/fitted.json results/nif/kodak" >> schedule.sh

# echo "echo 'Compressed' && python -m scripts.stats.dataset_compare results/nif/kodak/best/ results/nif/kodak/default/ compressed/stats.json compressed/stats.json" >> schedule.sh

# echo "python -m scripts.plots.self_compare" >> schedule.sh

