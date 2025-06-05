date 

FILE_PATH=$1
RESULTS_ROOT=$2
QUALITY=$3

WIDTH=$(identify -format "%w" $FILE_PATH)
HEIGHT=$(identify -format "%h" $FILE_PATH)

mkdir -p $RESULTS_ROOT

COMPRESSED_PATH=$RESULTS_ROOT/compressed.jxl
DECODED_PATH=$RESULTS_ROOT/decoded.png
STATS_PATH=$RESULTS_ROOT/stats.json

TMP_PATH=./.tmp.ppm

magick convert $FILE_PATH $TMP_PATH
cjxl $TMP_PATH $COMPRESSED_PATH -q $QUALITY 
rm $TMP_PATH
djxl $COMPRESSED_PATH $TMP_PATH
magick convert $TMP_PATH $DECODED_PATH
rm $TMP_PATH

python3 filewise_export_stats.py \
    $FILE_PATH \
    $DECODED_PATH \
    $STATS_PATH \
    $COMPRESSED_PATH

date
