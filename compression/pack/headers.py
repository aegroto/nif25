def pack_metadata(stream, metadata):
    stream.append_struct("!H", metadata["height"])
    stream.append_struct("!H", metadata["width"])

def unpack_metadata(stream):
    metadata = dict()
    metadata["height"] = stream.pull_struct("!H", 2)
    metadata["width"] = stream.pull_struct("!H", 2)
    return metadata

def pack_config(stream, config):
    stream.append_struct("!H", config["max_symbol"])
    stream.append_struct("!f", config["bound"])
    stream.append_struct("!f", config["scale"])
    stream.append_struct("!f", config["zero"])

def unpack_config(stream):
    config = dict()
    config["mode"] = "uniform"
    config["max_symbol"] = stream.pull_struct("!H", 2)
    config["bound"] = stream.pull_struct("!f")
    config["scale"] = stream.pull_struct("!f")
    config["zero"] = stream.pull_struct("!f")
    config["rounding_mode"] = "round" 
    return config
