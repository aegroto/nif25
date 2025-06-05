> schedule.sh

if true;
then
    for config in $(find configurations/.staging/*.yaml)
    # for config in "configurations/.nif/clic2020/msssim/128_32_3.0_5_msssim_0.70.yaml" \
    #                 "configurations/.nif/clic2020/msssim/224_32_4.0_5_msssim_0.20.yaml" \
    #                 "configurations/.nif/clic2020/visual/128_32_3.0_5_visual_0.70.yaml" \
    #                 "configurations/.nif/clic2020/visual/224_32_4.0_5_visual_0.40.yaml"
    # for config in $(find configurations/.nif/clic2020/visual/224_32_4_0_5_psnr_0.20.yaml)
    # for config in "configurations/test.yaml"
    # for config in "configurations/default.yaml"
    do
        config_file=$(basename $config)
        config_id="${config_file%.yaml}"

        for file in test_images/clic2020/*.png
        # for file in "martyn-seddon-220"
        # for file in results/nif/clic2020_baseline/320/*
        do
            file="test_images/clic2020/${file}"

            basename=$(basename $file)
            i=${basename%.*}

            # log_file="logs/${config_id}_$i.txt"

            # echo "./experiment.sh $config test_images/clic2020/$i.png results/nif/clic2020/$config_id/$i > $log_file 2>&1" >> schedule.sh
            echo "./scripts/run/fit_experiment.sh $config test_images/clic2020/$i.png results/nif/clic2020/$config_id/$i > "logs/${config_id}_${i}_fitting.txt" 2>&1" >> schedule.sh
            # echo "./scripts/run/tuning_experiment.sh $config test_images/clic2020/$i.png results/nif/clic2020/$config_id/$i > "logs/${config_id}_${i}_tuning.txt" 2>&1" >> schedule.sh
            # echo "./scripts/run/quantization_experiment.sh $config test_images/clic2020/$i.png results/nif/clic2020/$config_id/$i > "logs/${config_id}_${i}_quantization.txt" 2>&1" >> schedule.sh
            # echo "./scripts/run/compression_experiment.sh $config test_images/clic2020/$i.png results/nif/clic2020/$config_id/$i > "logs/${config_id}_${i}_compression.txt" 2>&1" >> schedule.sh

            # for attempt in {1..5}
            # do
            #     echo "export NIF_SEED=${attempt}" >> schedule.sh
            #     echo "./scripts/run/full_experiment.sh \"$config\" test_images/clic2020/$i.png \"results/nif/clic2020/$config_id/${i}_${attempt}\" > "logs/${config_id}_${i}_${attempt}_full.txt" 2>&1" >> schedule.sh
            #     echo "./scripts/run/decode_only_experiment.sh \"$config\" test_images/clic2020/$i.png \"results/nif/clic2020/$config_id/${i}_${attempt}\" > "logs/${config_id}_${i}_${attempt}_decode_only.txt" 2>&1" >> schedule.sh
            # done
        done
    done
fi

if true;
then
    for config in $(find configurations/.tuning/*.yaml)
    do
        config_file=$(basename $config)
        config_id="${config_file%.yaml}"

        base_config_id=${config_id::-5}
        echo "cp -rL results/nif/clic2020/${base_config_id} results/nif/clic2020/${config_id}" >> schedule.sh

        for file in test_images/clic2020/*.png
        do
            file="test_images/clic2020/${file}"

            basename=$(basename $file)
            i=${basename%.*}

            log_file="logs/${config_id}_$i.txt"

            echo "./scripts/run/quantization_experiment.sh $config test_images/clic2020/$i.png results/nif/clic2020/$config_id/$i > "logs/${config_id}_${i}_quantization.txt" 2>&1" >> schedule.sh
            echo "./scripts/run/compression_experiment.sh $config test_images/clic2020/$i.png results/nif/clic2020/$config_id/$i > "logs/${config_id}_${i}_compression.txt" 2>&1" >> schedule.sh
        done
    done
fi

echo "python calculate_summary.py compressed/stats.json results/summaries/clic2020/tmp/compressed.json results/nif/clic2020" >> schedule.sh
echo "python calculate_summary.py fitted/stats.json results/summaries/clic2020/tmp/fitted.json results/nif/clic2020" >> schedule.sh
echo "python -m scripts.plots.clic2020" >> schedule.sh 
echo "echo 'Compressed' && python -m scripts.stats.dataset_compare results/nif/clic2020/baseline/ results/nif/clic2020/default/ compressed/stats.json compressed/stats.json" >> schedule.sh 
