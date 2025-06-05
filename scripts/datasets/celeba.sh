> schedule.sh

# for config in $(find configurations/.nif/celeba/*.yaml)
# for config in $(find configurations/.nif/celeba/120.yaml)
for config in $(find configurations/default.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for file in test_images/celeba/*.png
    # for i in "182664" "185277" "190719" "200044" "202322"
    # for i in "182664"
    do
        basename=$(basename $file)
        i=${basename%.*}

        log_file="logs/${config_id}_$i.txt"
        echo "./experiment.sh $config test_images/celeba/$i.png results/nif/celeba/$config_id/$i > $log_file 2>&1" >> schedule.sh
    done
done

# echo "python3 calculate_summary.py stats.json results/summaries/celeba/nif.json results/nif/celeba/" >> schedule.sh
# echo "python3 plot_results.py results/summaries/celeba/ results.png psnr" >> schedule.sh
