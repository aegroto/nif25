> schedule.sh

# for config in $(find configurations/.staging/*.yaml)
for config in $(find configurations/.tuning/*.yaml)
# for config in $(find configurations/.nif/clic2024/*.yaml)
# for config in "configurations/default.yaml"
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for file in test_images/clic2024/*.png
    # for file in \
    #     "0f89df7638bc485db94a981367ad6983bda3396e153893cf80794790e48d3df7.png" \
    #     "2ff7069b3e9ba2e7a1aaf400783004d0dfcc762cdab00b8be922fe6b685d85ea.png" \
    #     "03bcfef063be6a7db416b1cf8c227f201d6a6b7c2aaee7200a46c96d2b4c4f37.png"
    do
        basename=$(basename $file)
        i=${basename%.*}

        log_file="logs/${config_id}_$i.txt"
        echo "./experiment.sh $config test_images/clic2024/$i.png results/nif/clic2024/$config_id/$i > $log_file 2>&1" >> schedule.sh
        # echo "./speed_experiment.sh $config test_images/clic2024/$i.png results/nif/clic2024/$config_id/$i > $log_file 2>&1" >> schedule.sh
        # echo "./quantization_experiment.sh $config $config_id test_images/clic2024/$i.png results/nif/clic2024/default/$i > $log_file 2>&1" >> schedule.sh
    done
done

# echo "python3 calculate_summary.py stats.json results/summaries/clic2024/nif.json results/nif/clic2024/" >> schedule.sh
# echo "python3 plot_results.py results/summaries/clic2024/ results.png psnr" >> schedule.sh
