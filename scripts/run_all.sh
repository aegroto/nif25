FOLDER=$2
for file in $FOLDER/*
do
    experiment_id=$(basename $file)
    # screen -d -m -S "$experiment_id" -L -Logfile "logs/$experiment_id.txt" python3 $1 $file
    python3 $1 $file
done

