import torch
from torchvision.models import vgg19

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained = True).features)
        self.features = torch.nn.ModuleList(features).eval()
        
    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results