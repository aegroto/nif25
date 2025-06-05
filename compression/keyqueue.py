from models.nif import NIF
from models.nif.utils import load_image_data_into_params

def key_queue_for_config(config, metadata):
    params = config["model"]
    params = load_image_data_into_params(params, metadata["height"], metadata["width"])
    model = NIF(**params)
    return list(model.state_dict().keys())
