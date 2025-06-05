DATASET=$1
QUALITY=$2

for file in test_images/$DATASET/*.png;
do
    image_filename=$(basename $file)
    image_id=${image_filename%.*}
    echo "Encoding $image_id with level $QUALITY..."
    ./scripts/bpg/run.sh $file results/bpg/$DATASET/$QUALITY/$image_id/ $QUALITY
done
