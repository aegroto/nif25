import copy
import numpy

import torch
import sys
from compression.pack.consts import is_no_quant
from compression.pack.headers import unpack_config, unpack_metadata
from compression.quantization import quantizer_from_config

from compression.bytestream import ByteStream
from compression.keyqueue import key_queue_for_config
from compression.quantization.uniform import UniformQuantizer
from compression.utils import numpy_type_for_bits
from serialization import deserialize_state_dict
from utils import load_configuration, load_device

def decompress_state_dict(config_path, compressed_state_dict, device=None):
    model_config = load_configuration(config_path)

    print("Decompressing...")
    coded_stream = ByteStream(compressed_state_dict)

    decompressed_state_dict = dict()

    metadata = unpack_metadata(coded_stream)

    for key in key_queue_for_config(model_config, metadata):
        quantization_config = unpack_config(coded_stream)

        if is_no_quant(quantization_config):
            packed_result = coded_stream.pull_bytes()
            array = numpy.frombuffer(packed_result, numpy.float32).copy()
            dequantized_tensor = torch.from_numpy(array).to(device)
        else:
            config = copy.deepcopy(quantization_config)
            config["device"] = device
            quantizer = quantizer_from_config(config)

            quantization_result = quantizer.packer.unpack(coded_stream)

            dequantized_tensor = quantizer.dequantize_tensor(quantization_result)

        decompressed_state_dict[key] = dequantized_tensor

    decompressed_state_dict["__meta"] = metadata

    return decompressed_state_dict

if __name__ == "__main__":
    compressed_state_dict_path = sys.argv[1]
    decompressed_state_dict_path = sys.argv[2]

    print("Loading compressed state dict...")
    # compressed_state_dict = torch.load(compressed_state_dict_path)
    compressed_state_dict = deserialize_state_dict(compressed_state_dict_path)

    decompressed_state_dict = decompress_state_dict(compressed_state_dict, device=load_device())

    torch.save(decompressed_state_dict, decompressed_state_dict_path)

