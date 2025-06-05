> schedule.sh

# for config in $(find configurations/.nif/icb/*.yaml)
for config in "configurations/default.yaml"
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for file in test_images/icb/*.png
    # for file in "test_images/icb/deer.png" "test_images/icb/bridge.png"
    do
        basename=$(basename $file)
        i=${basename%.*}

        log_file="logs/${config_id}_$i.txt"
        # echo "./experiment.sh $config test_images/icb/$i.png results/nif/icb/$config_id/$i > $log_file 2>&1" >> schedule.sh
        echo "./speed_experiment.sh $config test_images/icb/$i.png results/nif/icb/$config_id/$i > $log_file 2>&1" >> schedule.sh
    done
done

# echo "python3 calculate_summary.py stats.json results/summaries/icb/nif.json results/nif/icb/" >> schedule.sh
# echo "python3 plot_results.py results/summaries/icb/ results.png psnr" >> schedule.sh
