DATASET=$1
QUALITY=$2

for file in test_images/$DATASET/*.png;
do
    image_filename=$(basename $file)
    image_id=${image_filename%.*}
    echo "Encoding $image_id with quality $QUALITY..."
    ./scripts/jxl/run.sh $file results/jxl/$DATASET/$QUALITY/$image_id/ $FORMAT $QUALITY
done
