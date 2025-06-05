def indexed_keys(config):
    return enumerate(config.keys())

def extract_max_symbols(config):
    result = dict()
    for (_, key) in indexed_keys(config):
        result[key] = config[key]["max_symbol"]

    return result

def apply_max_symbols_in_config(max_symbols, config):
    for (idx, key) in indexed_keys(config):
        config[key]["max_symbol"] = max_symbols[key]

    return config

def load_parameters_stats(config, model):
    stats = dict()
    state_dict = model.state_dict()
    for (_, key) in indexed_keys(config):
        stats[key] = {
            "num_elements": state_dict[f"{key}._values"].numel()
        }
    return stats


