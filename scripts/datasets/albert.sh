> schedule.sh

for config in $(find configurations/.tuning/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    log_file="logs/${config_id}.txt"
    echo "./experiment.sh $config test_images/albert.png results/nif/albert/$config_id > $log_file 2>&1" >> schedule.sh
done

# echo "python3 calculate_summary.py stats.json results/summaries/kodak/nif.json results/nif/kodak/" >> schedule.sh
# echo "python3 plot_results.py results/summaries/kodak/ results.png psnr" >> schedule.sh
