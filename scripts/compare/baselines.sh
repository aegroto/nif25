ORIGINAL=$1
UPSAMPLED=$2

METRIC_PARAMS="$ORIGINAL $UPSAMPLED /dev/null"
MAE=$(magick compare -metric MAE $METRIC_PARAMS 2>&1)
MSE=$(magick compare -metric MSE $METRIC_PARAMS 2>&1)
PSNR=$(magick compare -metric PSNR $METRIC_PARAMS 2>&1)
SSIM=$(magick compare -metric SSIM $METRIC_PARAMS 2>&1)

echo "$BASELINE_ID,$MAE,$MSE,$PSNR,$SSIM" >> $STATS_FILE
