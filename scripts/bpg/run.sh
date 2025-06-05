INPUT_PATH=$1
OUTPUT_FOLDER=$2
QUALITY=$3

mkdir -p $OUTPUT_FOLDER 

ABS_INPUT_PATH=$(readlink -f $INPUT_PATH)
ABS_OUTPUT_FOLDER=$(readlink -f $OUTPUT_FOLDER)

docker run -it --entrypoint bpgenc \
    -v $ABS_INPUT_PATH:/bpg/data/input.png \
    -v $ABS_OUTPUT_FOLDER:/bpg/data/output/ \
    ideaplexus/bpg -q $QUALITY /bpg/data/input.png -o /bpg/data/output/compressed.bpg  

docker run -it --entrypoint bpgdec \
    -v $ABS_OUTPUT_FOLDER:/bpg/data/ \
    ideaplexus/bpg -o decoded.png compressed.bpg

python3 filewise_export_stats.py \
    $INPUT_PATH \
    $OUTPUT_FOLDER/decoded.png \
    $OUTPUT_FOLDER/stats.json \
    $OUTPUT_FOLDER/compressed.bpg
