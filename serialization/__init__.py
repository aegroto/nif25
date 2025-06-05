import pickle
import sys
import struct
import torch

def save_buffer_to_file(buffer, path):
    with open(path, "wb") as file:
        file.write(buffer.getbuffer())

def serialize_state_dict(state_dict, path):
    data = pickle.dumps(state_dict)
    write_serialized_state_dict(data, path)

def write_serialized_state_dict(data, path):
    with open(path, "wb") as file:
        file.write(data)
        # pickle.dump(state_dict, file)

def deserialize_state_dict(path):
    with open(path, "rb") as file:
        return pickle.load(file)

def read_serialized_state_dict(path):
    with open(path, "rb") as file:
        return file.read()
