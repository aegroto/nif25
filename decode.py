import argparse
import time
from skimage import io
import json
from decompress import decompress_state_dict
from filewise_export_stats import export_stats
from infer import infer

from serialization import deserialize_state_dict, read_serialized_state_dict
from utils import load_device, set_reproducibility, setup_logging

def main():
    setup_logging()
    set_reproducibility()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("compressed_path", type=str)
    parser.add_argument("decoded_path", type=str)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--original_file_path", type=str, default=None)
    parser.add_argument("--time_stats_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    compressed_state_dict = read_serialized_state_dict(args.compressed_path)
    decompressed_state_dict = decompress_state_dict(args.config_path, compressed_state_dict, device=load_device())

    reconstructed_image = infer(args.config_path, decompressed_state_dict)
    io.imsave(args.decoded_path, reconstructed_image)
    end_time = time.time()
    total_time = end_time - start_time

    if args.stats_path is not None:
        export_stats(args.original_file_path, args.decoded_path, args.stats_path, args.compressed_path)

    if args.time_stats_path is not None:
        time_stats = json.load(open(args.time_stats_path, "r"))
        time_stats["decode_time_no_context"] = total_time
        json.dump(time_stats, open(args.time_stats_path, "w"))

if __name__ == "__main__":
    main()
