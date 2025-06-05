# for quality in 10 20 30 40 50 60 70 80 90
for quality in 30 50 70
do
    ./scripts/magick/dataset_run.sh clic2024 avif $quality
done
