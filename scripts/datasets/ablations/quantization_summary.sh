for bits in {4..16}
do 
    metric="psnr"
    show_metric="PSNR"
    bits_with_leading=$(printf "%02d" $bits)

    # mkdir results/nif/ablations/kodak/${metric}_${bits_with_leading}bits/
    # cp -r results/nif24/kodak_ablations/quantization/${metric}/${metric}_*_${bits_with_leading}bits results/nif/ablations/kodak/${metric}_${bits_with_leading}bits/
    # continue

    python calculate_summary.py compressed/stats.json \
        results/summaries/ablations/kodak/quantization/${metric}_${bits_with_leading}bits.json \
        results/nif/ablations/kodak/${metric}_${bits_with_leading}bits/ \
        --codec_name "NIF25 [${show_metric}] - ${bits}-bits quantization"
done
