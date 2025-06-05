IMAGE_ID=$1
CROP_W=$2
CROP_H=$3
CROP_X=$4
CROP_Y=$5

CROP_PARAMS="${CROP_W}x${CROP_H}+${CROP_X}+${CROP_Y}"
CROP_RECTANGLE="${CROP_X},${CROP_Y} $((${CROP_W}+${CROP_X})),$((${CROP_H}+${CROP_Y}))"

DRAW_PARAMS="-stroke white -strokewidth 3 -fill transparent" 

RESULTS_PATH="visual_comparisons/kodak_$IMAGE_ID"
STATS_PATH="visual_comparisons/kodak_$IMAGE_ID/stats/"

rm -r $RESULTS_PATH
mkdir -p $RESULTS_PATH/full
mkdir -p $RESULTS_PATH/crop
mkdir -p $STATS_PATH

magick "test_images/kodak/$IMAGE_ID.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/ground_truth.png"

magick "results/nif/kodak/$NIF_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/nif.png"
cp "results/nif/kodak/$NIF_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/nif.json"

magick "results/nif/losses/kodak/l1/$NIF_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/l1.png"
cp "results/nif/losses/kodak/l1/$NIF_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/l1.json"

magick "results/nif/losses/kodak/mse/$NIF_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/mse.png"
cp "results/nif/losses/kodak/mse/$NIF_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/mse.json"

magick "results/nif/losses/kodak/l1_ssim/$NIF_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/l1_ssim.png"
cp "results/nif/losses/kodak/l1_ssim/$NIF_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/l1_ssim.json"

magick "results/nif/losses/kodak/mse_ssim/$NIF_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/mse_ssim.png"
cp "results/nif/losses/kodak/mse_ssim/$NIF_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/mse_ssim.json"

magick "test_images/kodak/$IMAGE_ID.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/ground_truth.png"
magick "results/nif/kodak/$NIF_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/nif.png"
magick "results/nif/losses/kodak/l1/$NIF_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/l1.png"
magick "results/nif/losses/kodak/mse/$NIF_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/mse.png"
magick "results/nif/losses/kodak/l1_ssim/$NIF_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/l1_ssim.png"
magick "results/nif/losses/kodak/mse_ssim/$NIF_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/mse_ssim.png"
