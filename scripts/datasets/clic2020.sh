# > schedule.sh

# for preset in "psnr" "msssim" "visual"
for preset in "visual"
    do
    # for config in $(find configurations/.nif/clic2020/${preset}/*.yaml)
    # for config_name in "272_48_4.5_6_psnr_0.40" "320_80_5.0_5_psnr_0.50" "368_96_5.0_5_psnr_0.70"
    # for config_name in "272_48_4.5_6_msssim_0.30" "320_80_5.0_5_msssim_0.10" "368_96_5.0_5_msssim_0.10"
    for config_name in "272_48_4.5_6_visual_0.30" "320_80_5.0_5_visual_0.10" "368_96_5.0_5_visual_0.10"
    do
        config="configurations/.nif/clic2020/${preset}/${config_name}.yaml"

        config_file=$(basename $config)
        config_id="${config_file%.yaml}"

        for file in test_images/clic2020/*.png
        do
            basename=$(basename $file)
            i=${basename%.*}

            log_file="logs/${config_id}_$i.txt"
            echo "./scripts/run/full_experiment.sh \"$config\" test_images/clic2020/$i.png \"results/nif/clic2020/$config_id/$i\" > "logs/${config_id}_${i}_full.txt" 2>&1" >> schedule.sh
        done
    done
done
