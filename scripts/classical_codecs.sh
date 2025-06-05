# General parameters
SOURCE_IMAGE=$1
export FOLDER=$2
export DOWNSAMPLING_FACTOR=$3

rm -rf $FOLDER
mkdir -p $FOLDER
export RAW_IMAGE="$FOLDER/raw.rgb"

export WIDTH=$(identify -format "%w" $SOURCE_IMAGE)
export HEIGHT=$(identify -format "%h" $SOURCE_IMAGE)

magick convert $SOURCE_IMAGE $RAW_IMAGE

mkdir -p $FOLDER

export STATS_FILE="$FOLDER/stats.csv"
touch $STATS_FILE

echo "format,quality,bitrate,mae,mse,psnr,ssim" > $STATS_FILE

### Encodings
for FORMAT in "png" "jpeg" "webp" 
    do
    for QUALITY in "10" "50" "90" 
    do
        echo "Encoding in $FORMAT with quality $QUALITY..."
        FOLDER="$FOLDER/${FORMAT}_$QUALITY/" ./scripts/encode/convert.sh $FORMAT $QUALITY
    done
done

# rm $RAW_IMAGE
