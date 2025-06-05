DATASET=$1
> schedule.sh

for config in $(find configurations/.nif/${DATASET}_speed/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    echo "echo %%% Start $DATASET $config_id" >> schedule.sh
    echo "date" >> schedule.sh
    for file in test_images/${DATASET}/*.png
    do
        basename=$(basename $file)
        i=${basename%.*}

        log_file="logs/${config_id}_$i.txt"
        echo "./experiment.sh $config $file results/nif/${DATASET}/$config_id/$i > $log_file 2>&1" >> schedule.sh
    done
    echo "echo %%% End $DATASET $config_id" >> schedule.sh
    echo "date" >> schedule.sh
done

# echo "python3 calculate_summary.py stats.json results/summaries/${DATASET}_speed/nif.json results/nif/${DATASET}_speed/" >> schedule.sh
# echo "python3 plot_results.py results/summaries/${DATASET}_speed/ results.png psnr" >> schedule.sh
