import math
import constriction
import brotli
import torch
import numpy
import pickle
import logging
from compression.bytestream import ByteStream

from compression.utils import numpy_type_for_bits, torch_type_for_bits

LOGGER = logging.getLogger(__name__)

class BrotliEncoder:
    def __init__(self, bits):
        self.torch_type = torch_type_for_bits(bits)
        self.numpy_type = numpy_type_for_bits(bits)

    def compress(self, tensor):
        return brotli.compress(tensor.cpu().to(self.torch_type).numpy().astype(self.numpy_type).tobytes())

    def decompress(self, compressed):
        array = numpy.frombuffer(brotli.decompress(compressed), dtype=self.numpy_type).copy()
        return torch.from_numpy(array).to(torch.float32)

def build_gaussian_entropy_model(min_symbol, max_symbol, mean, std):
        scale = std
        return constriction.stream.model.QuantizedGaussian(
            min_symbol, max_symbol, mean, scale)

def build_laplace_entropy_model(min_symbol, max_symbol, mean, std):
        scale = std / math.sqrt(2.0)
        return constriction.stream.model.QuantizedLaplace(
            min_symbol, max_symbol, mean, scale)

class RangeEncoder:
    def __init__(self, model_builder):
        self.model_builder = model_builder

    def compress(self, tensor):
        array = tensor.cpu().to(torch.int32).numpy()

        stream = ByteStream()

        symbols = array.flatten()

        num_symbols = len(symbols)

        encoder = constriction.stream.queue.RangeEncoder()
        min_symbol = numpy.min(symbols)
        max_symbol = numpy.max(symbols)
        mean = numpy.mean(symbols).astype(numpy.float16)
        std = numpy.std(symbols).astype(numpy.float16)

        entropy_model = self.model_builder(min_symbol, max_symbol, mean, std)

        encoder.encode(symbols, entropy_model)

        buffer = encoder.get_compressed().tobytes()

        LOGGER.debug(f"Buffer len: {len(buffer)}")

        stream.append_bytes(buffer)
        stream.append_struct("!h", min_symbol)
        stream.append_struct("!h", max_symbol)
        stream.append_struct("!e", mean)
        stream.append_struct("!e", std)
        stream.append_struct("!I", num_symbols)

        return stream.data()

    def decompress(self, result):
        stream_data = result
        stream = ByteStream(stream_data)

        buffer = stream.pull_bytes()
        min_symbol = stream.pull_struct("!h", 2)
        max_symbol = stream.pull_struct("!h", 2)
        mean = stream.pull_struct("!e", 2)
        std = stream.pull_struct("!e", 2)
        num_symbols = stream.pull_struct("!I")

        entropy_model = self.model_builder(min_symbol, max_symbol, mean, std)
        compressed = numpy.frombuffer(buffer, dtype=numpy.uint32)

        decoder = constriction.stream.queue.RangeDecoder(compressed)

        symbols = decoder.decode(entropy_model, num_symbols)

        array = symbols

        return torch.from_numpy(array).to(torch.float32)

class FeaturewiseRangeEncoder:
    def __init__(self):
        self.__inner_encoder = RangeEncoder(build_laplace_entropy_model)
        self.__dim = 0

    def compress(self, tensor):
        # tensor = tensor.flatten().reshape(tensor.numel() // 4, 4)
        feature_tensors = tensor.unbind(self.__dim)

        compressed_features = [
            self.__inner_encoder.compress(feature_tensor) for feature_tensor in feature_tensors
        ]

        buffer_lens = list()
        for compressed_feature in compressed_features:
            (buffer, min_symbol, max_symbol, mean, std, num_symbols) = pickle.loads(compressed_feature)
            buffer_lens.append(len(buffer))

        LOGGER.debug(f"Total buffer len: {numpy.sum(buffer_lens)}")

        return pickle.dumps(compressed_features)

    def decompress(self, result):
        compressed_features = pickle.loads(result)

        feature_tensors = [
            self.__inner_encoder.decompress(compressed_feature).unsqueeze(self.__dim) for compressed_feature in compressed_features
        ]

        tensor = torch.cat(feature_tensors, 0)
        tensor = tensor.flatten()
        return tensor