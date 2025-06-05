import torch

from utils import load_device 

class CompositeLoss:
    def __init__(self, losses, return_components=False):
        self.__losses = losses
        self.__return_components = return_components
    
    def __len__(self):
        return len(self.__losses)

    def __call__(self, *args):
        total_loss = torch.zeros(1, device=load_device())
        components = list()
        for (idx, component) in enumerate(self.__losses):
            (loss_fn, delta) = component
            loss = loss_fn(*args)
            total_loss += loss * delta
            components.append((idx, type(loss_fn).__name__, loss))

        if self.__return_components:
            return total_loss, components
        else:
            return total_loss


