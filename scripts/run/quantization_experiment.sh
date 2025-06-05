date 

CONFIGURATION_PATH=$1
FILE_PATH=$2
# INPUT_ROOT=$3
RESULTS_ROOT=$3

# TUNED_PATH=$RESULTS_ROOT/tuned/state.pth
TUNED_PATH=$RESULTS_ROOT/fitted/state.pth

mkdir -p $RESULTS_ROOT
mkdir -p $RESULTS_ROOT/quantized/

QUANTIZED_PATH=$RESULTS_ROOT/quantized/quantized.pth
FP_QUANTIZED_PATH=$RESULTS_ROOT/quantized/fp_quantized.pth

DECODED_PATH=$RESULTS_ROOT/quantized/decoded.png
STATS_PATH=$RESULTS_ROOT/quantized/stats.json

python3 quantize.py $CONFIGURATION_PATH $FILE_PATH $TUNED_PATH $QUANTIZED_PATH $FP_QUANTIZED_PATH --infer_path $DECODED_PATH --stats_path $STATS_PATH

date
