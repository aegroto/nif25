DATASET=$1
FORMAT=$2
QUALITY=$3

for file in test_images/$DATASET/*.png;
do
    image_filename=$(basename $file)
    image_id=${image_filename%.*}
    echo "Encoding $image_id with quality $QUALITY..."
    ./magick_experiment.sh $file results/$FORMAT/$DATASET/$QUALITY/$image_id/ $FORMAT $QUALITY
done
