import re
import torch

class ModelLoss():
    def __init__(self, regex = None):
        self.__parameters = None
        self.__max_size = 0
        self.__regex = regex

    def __load_modules(self, model):
        self.__parameters = list()
        for name, parameter in model.named_parameters():
            if self.__regex is not None and not re.search(self.__regex, name):
                continue

            if parameter.requires_grad:
                self.__parameters.append(parameter)
                size = parameter.numel()
                self.__max_size = max(size, self.__max_size)

    def _max_size(self):
        return self.__max_size

    def _call(self):
        pass

    def __call__(self, model):
        if self.__parameters is None:
            self.__load_modules(model)

        return self._call(self.__parameters)

class QuantizerLoss():
    def __init__(self, regex = None):
        self.__modules = None
        self.__max_size = 0
        self.__regex = regex

    def __load_modules(self, model):
        self.__modules = list()
        for name, module in model.named_modules():
            if not hasattr(module, "quantizer"):
                continue

            if self.__regex is not None and not re.search(self.__regex, name):
                continue
        
            self.__modules.append(module)
            size = module._values.numel()
            self.__max_size = max(size, self.__max_size)

    def _max_size(self):
        return self.__max_size

    def _call(self):
        pass

    def __call__(self, model):
        if self.__modules is None:
            self.__load_modules(model)

        return self._call(self.__modules)