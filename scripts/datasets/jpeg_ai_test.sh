> schedule.sh

for preset in "psnr" "msssim" "visual"; do
    for config in $(find configurations/.nif/jpeg_ai_test/${preset}/*.yaml); do
        config_file=$(basename $config)
        config_id="${config_file%.yaml}"

        for file in test_images/jpeg_ai_test/*.png
        do
            basename=$(basename $file)
            i=${basename%.*}

            log_file="logs/${config_id}_$i.txt"
            echo "./scripts/run/full_experiment.sh \"$config\" test_images/jpeg_ai_test/$i.png \"results/nif/jpeg_ai_test/$config_id/$i\" > "logs/${config_id}_${i}_full.txt" 2>&1" >> schedule.sh
        done
    done
done

