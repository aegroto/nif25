import torch
import argparse
from compress import compress_state_dict

from fit import fit
from quantize import quantize
from serialization import serialize_state_dict, write_serialized_state_dict
from utils import set_reproducibility

def main():
    set_reproducibility()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("image_path", type=str)
    parser.add_argument("compressed_path", type=str)
    parser.add_argument("--uncompressed_state_path", type=str, required=False)

    args = parser.parse_args()

    uncompressed_state_dict = fit(args.config_path, args.image_path)
    fp_quantized_state_dict, _ = quantize(args.config_path, args.image_path, uncompressed_state_dict)
    if args.uncompressed_state_path:
        serialize_state_dict(fp_quantized_state_dict, args.uncompressed_state_path)

    compressed_state_dict = compress_state_dict(args.config_path, fp_quantized_state_dict)

    write_serialized_state_dict(compressed_state_dict, args.compressed_path)

if __name__ == "__main__":
    main()
