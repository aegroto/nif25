date 

CONFIGURATION_PATH=$1
FILE_PATH=$2
RESULTS_ROOT=$3

mkdir -p $RESULTS_ROOT/fitted

STATE_PATH=$RESULTS_ROOT/fitted/state.pth
OPTIMIZER_PATH=$RESULTS_ROOT/fitted/optimizer_state.pth
DECODED_PATH=$RESULTS_ROOT/fitted/decoded.png
STATS_PATH=$RESULTS_ROOT/fitted/stats.json

python3 fit.py $CONFIGURATION_PATH $FILE_PATH $STATE_PATH $OPTIMIZER_PATH --infer_path $DECODED_PATH --stats_path $STATS_PATH

date
