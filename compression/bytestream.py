import struct
import pickle

class ByteStream:
    def __init__(self, initial_data=None):
        if initial_data is None:
            initial_data = bytes()

        self.__data = initial_data
        self.__current_index = 0

    def __write_data(self, new_data):
        self.__data += new_data

    def __read_data(self, number):
        data = self.__data[self.__current_index:self.__current_index+number]
        self.__current_index += number
        return data

    def append_bytes(self, new_data):
        self.__write_data(int.to_bytes(len(new_data), 4, "big"))
        self.__write_data(new_data)

    def pull_bytes(self):
        length = int.from_bytes(self.__read_data(4), "big")
        data = self.__read_data(length)
        return data

    def append_obj(self, obj):
        serialized_obj = pickle.dumps(obj)
        self.append_bytes(serialized_obj)

    def pull_obj(self):
        serialized_obj = self.pull_bytes()
        obj = pickle.loads(serialized_obj)
        return obj

    def append_string(self, value):
        str_bytes = str.encode(value)
        self.append_bytes(str_bytes)

    def pull_string(self):
        str_bytes = self.pull_bytes()
        return str_bytes.decode()

    def append_struct(self, format, value):
        serialized_value = struct.pack(format, value)
        self.__write_data(serialized_value)

    def pull_struct(self, format, size=4):
        serialized_value = self.__read_data(size)
        value = struct.unpack(format, serialized_value)[0]
        return value

    def is_empty(self):
        return self.__current_index >= len(self.__data)

    def data(self):
        return self.__data

    def len(self):
        return len(self.__data)
