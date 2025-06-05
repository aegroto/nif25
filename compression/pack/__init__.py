import math
import pickle
import logging

from compression.pack.encoders import BrotliEncoder, RangeEncoder, build_gaussian_entropy_model, build_laplace_entropy_model

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

class UniformQuantizedPacker():
    def __init__(self, max_symbol, device):
        self.bits = math.ceil(math.log2(max_symbol + 1)) + 1 # bits
        self.device = device

        self.encoders = [
            RangeEncoder(build_laplace_entropy_model),
            RangeEncoder(build_gaussian_entropy_model),
            BrotliEncoder(self.bits)
        ]

    def pack(self, stream, result):
        quantized_tensor = result

        LOGGER.debug(" ---- ")

        best_idx = -1
        best_size = -1
        for (idx, encoder) in enumerate(self.encoders):
            compressed = encoder.compress(quantized_tensor)
            size = len(compressed)
            LOGGER.debug(f"#[{idx}] {type(encoder).__name__} compressed: {size}")
            if best_idx == -1 or best_size > size:
                best_idx = idx  
                best_size = size

        quantized_compressed = self.encoders[best_idx].compress(quantized_tensor)

        LOGGER.debug(f"Selected: {best_idx}")

        stream.append_struct("!B", best_idx)
        stream.append_bytes(quantized_compressed)

    def unpack(self, stream):
        best_idx = stream.pull_struct("!B", 1)
        quantized_compressed = stream.pull_bytes()

        LOGGER.debug(f"Decoding with encoder {best_idx}")

        quantized_tensor = self.encoders[best_idx].decompress(quantized_compressed).to(self.device)

        return quantized_tensor
