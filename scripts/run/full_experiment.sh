date 

CONFIGURATION_PATH=$1
FILE_PATH=$2
RESULTS_ROOT=$3

mkdir -p $RESULTS_ROOT

encode_start=`date +%s%N`
python -m encode \
    $CONFIGURATION_PATH \
    $FILE_PATH \
    $RESULTS_ROOT/compressed.nif
encode_end=`date +%s%N`
encode_time=`expr $encode_end - $encode_start`
encode_time_secs=$(bc <<< "scale=4; $encode_time/1000000000")
echo "Encode time: ${encode_time_secs}s"

decode_start=`date +%s%N`
python -m decode \
    $CONFIGURATION_PATH \
    $RESULTS_ROOT/compressed.nif \
    $RESULTS_ROOT/decoded.png \
    --stats_path $RESULTS_ROOT/stats.json \
    --original_file_path $FILE_PATH
decode_end=`date +%s%N`
decode_time=`expr $decode_end - $decode_start`
decode_time_secs=$(bc <<< "scale=4; $decode_time/1000000000")
echo "Decode time: ${decode_time_secs}s"

echo "{ \"encode_time\": ${encode_time_secs}, \"decode_time\": ${decode_time_secs} }" > $RESULTS_ROOT/times.json
