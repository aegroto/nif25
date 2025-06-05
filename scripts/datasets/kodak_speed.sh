> schedule.sh

for config in $(find configurations/.nif/kodak_speed/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.toml}"
    for i in 24 3 7 8 20
    do
        log_file="logs/${config_id}_$i.txt"
        echo "./experiment.sh $config test_images/kodak/$i.png results/nif/kodak_speed/$config_id/$i > $log_file 2>&1" >> schedule.sh
    done
done

# echo "python3 calculate_summary.py stats.json results/summaries/kodak/nif.json results/nif/kodak/" >> schedule.sh
# echo "python3 plot_results.py results/summaries/kodak/ results.png psnr" >> schedule.sh
