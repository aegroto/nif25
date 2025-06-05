date 

CONFIGURATION_PATH=$1
FILE_PATH=$2
RESULTS_ROOT=$3

FP_QUANTIZED_PATH=$RESULTS_ROOT/quantized/fp_quantized.pth

mkdir -p $RESULTS_ROOT/compressed

COMPRESSED_PATH=$RESULTS_ROOT/compressed/compressed.nif
DECODED_PATH=$RESULTS_ROOT/compressed/decoded.png
STATS_PATH=$RESULTS_ROOT/compressed/stats.json

python3 compress.py $CONFIGURATION_PATH $FP_QUANTIZED_PATH $COMPRESSED_PATH
python3 decode.py $CONFIGURATION_PATH $COMPRESSED_PATH $DECODED_PATH --stats_path $STATS_PATH --original_file_path $FILE_PATH

date
