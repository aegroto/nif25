# Setup
FORMAT=$1
QUALITY=$2

mkdir -p $FOLDER

# Encoding
COMPRESSED_IMAGE=$FOLDER/compressed.$FORMAT
magick convert -quality $QUALITY -depth 8 -size ${WIDTH}x${HEIGHT} $RAW_IMAGE $COMPRESSED_IMAGE

DOWNSAMPLED_COMPRESSED_IMAGE=$FOLDER/downsampled.$FORMAT
magick convert -quality $QUALITY -depth 8 -size ${WIDTH}x${HEIGHT} $RAW_IMAGE -resize $DOWNSAMPLING_FACTOR $DOWNSAMPLED_COMPRESSED_IMAGE

# Size estimation
COMPRESSED_SIZE=$(stat --printf="%s" "$FOLDER/compressed.$FORMAT")
SIZE_IN_BITS=$(($COMPRESSED_SIZE * 8))
PIXELS_COUNT=$(($WIDTH*$HEIGHT))
BPP=$(magick convert xc: -format "%[fx:$SIZE_IN_BITS/$PIXELS_COUNT]" info:)

METRIC_PARAMS="-depth 8 -size ${WIDTH}x${HEIGHT} $RAW_IMAGE $COMPRESSED_IMAGE /dev/null"
MAE=$(magick compare -metric MAE $METRIC_PARAMS 2>&1)
MSE=$(magick compare -metric MSE $METRIC_PARAMS 2>&1)
PSNR=$(magick compare -metric PSNR $METRIC_PARAMS 2>&1)
SSIM=$(magick compare -metric SSIM $METRIC_PARAMS 2>&1)

echo "$FORMAT,$QUALITY,$BPP,$MAE,$MSE,$PSNR,$SSIM" >> $STATS_FILE
