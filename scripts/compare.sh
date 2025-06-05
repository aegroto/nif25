# echo "Unpruned vs Pruned"
# echo "SSIM"
# echo $(magick compare -metric SSIM comparisons/bosphorus_test_x2/reconstructed_unpruned.png comparisons/bosphorus_test_x2/reconstructed.png /dev/null)
# echo "PSNR"
# echo $(magick compare -metric PSNR comparisons/bosphorus_test_x2/reconstructed_unpruned.png comparisons/bosphorus_test_x2/reconstructed.png /dev/null)

ORIGINAL="test_images/kodak/original/13.png"
UNQUANTIZED="unquantized.png"
DECODED="decoded.png"
DEQUANTIZED="dequantized.png"

echo "Original vs Unquantized"
echo $(magick compare -metric PSNR $ORIGINAL $UNQUANTIZED /dev/null)
echo "Original vs Decoded"
echo $(magick compare -metric PSNR $ORIGINAL $DECODED /dev/null)
echo "Original vs Dequantized"
echo $(magick compare -metric PSNR $ORIGINAL $DEQUANTIZED /dev/null)

