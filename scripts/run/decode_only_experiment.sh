date 

CONFIGURATION_PATH=$1
FILE_PATH=$2
RESULTS_ROOT=$3

python -m decode \
    $CONFIGURATION_PATH \
    $RESULTS_ROOT/compressed.nif \
    $RESULTS_ROOT/decoded.png \
    --original_file_path $FILE_PATH \
    --time_stats_path $RESULTS_ROOT/times.json \
