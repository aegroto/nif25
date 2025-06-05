import torch

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, epsilon=10.0e-3):
        super().__init__()

        self.penalty = torch.square(torch.tensor(epsilon)) 

    def __call__(self, img1, img2):
        diff = img1 - img2
        return torch.sqrt(torch.square(diff) + self.penalty).mean()

