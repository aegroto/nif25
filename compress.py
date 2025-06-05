import sys
import copy
import logging

import argparse
from compression.pack.consts import NO_QUANT_CONFIG
from compression.pack.headers import pack_config, pack_metadata
from compression.quantization import quantizer_from_config

from compression.bytestream import ByteStream
from compression.keyqueue import key_queue_for_config
from compression.quantization.uniform import UniformQuantizer
from serialization import deserialize_state_dict, write_serialized_state_dict
from utils import load_configuration, setup_logging


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def size(tensor):
    return tensor.element_size() * tensor.nelement()

def compress_state_dict(config_path, state_dict):
    config = load_configuration(config_path)

    quantization_config = state_dict["quantization_config"]
    del state_dict["quantization_config"]

    metadata = copy.deepcopy(state_dict["__meta"])
    del state_dict["__meta"]

    LOGGER.debug("Compressing...")

    coded_stream = ByteStream()

    len_before = coded_stream.len()
    pack_metadata(coded_stream, metadata)
    LOGGER.debug(f"Metadata weight: {coded_stream.len() - len_before}")

    for key in key_queue_for_config(config, metadata):
        tensor = state_dict[key]

        len_before = coded_stream.len()

        if key not in quantization_config:
            LOGGER.debug(f"No quantization config found for {key}, assuming default config")
            quantization_result = tensor

            pack_config(coded_stream, NO_QUANT_CONFIG)
            packed_result = quantization_result.cpu().numpy().tobytes()
            coded_stream.append_bytes(packed_result)
        else:
            quantizer_config = copy.deepcopy(quantization_config[key])
            quantizer_config["device"] = tensor.device
            quantizer = quantizer_from_config(quantizer_config)
            quantization_result = quantizer.quantize_tensor(tensor)

            updated_quantizer_config = quantizer.get_config()

            pack_config(coded_stream, updated_quantizer_config)
            quantizer.packer.pack(coded_stream, quantization_result)
            LOGGER.debug(f"{key} weight: {coded_stream.len() - len_before}")

    compressed_state_dict = coded_stream.data()

    return compressed_state_dict

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("model_dump_path", type=str)
    parser.add_argument("compressed_model_dump_path", type=str)
    args = parser.parse_args()

    LOGGER.debug("Loading state dict...")
    state_dict = deserialize_state_dict(args.model_dump_path)
    compressed_state_dict = compress_state_dict(args.config_path, state_dict)

    write_serialized_state_dict(compressed_state_dict, args.compressed_model_dump_path)
