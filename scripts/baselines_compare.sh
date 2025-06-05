ORIGINAL_FOLDER=$1
COMPARISONS_FOLDER=$2

for file in $ORIGINAL_FOLDER/*
do
    filename=$(basename $file)
    image_id=${filename%.*}
    echo $image_id


    original_file=$file
    upsampled_folder="$COMPARISONS_FOLDER/$image_id/baselines/"

    export STATS_FILE="$COMPARISONS_FOLDER/$image_id/baseline_stats.csv"
    touch $STATS_FILE
    echo "id,mae,mse,psnr,ssim" > $STATS_FILE

    for upsampled_file in $upsampled_folder/*
    do
        upsampled_filename=$(basename $upsampled_file)
        export BASELINE_ID=${upsampled_filename%.*}
        ./scripts/compare/baselines.sh $original_file $upsampled_file
    done
done
