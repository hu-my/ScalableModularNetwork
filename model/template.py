import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import OneConvEncoder, TwoConvEncoder
from abc import abstractmethod

class Template(nn.Module):
    def __init__(self, args):
        super(Template, self).__init__()
        self.args = args
        if args['cuda']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if args['encoder'] == 'oneconv':
            self.encoder = OneConvEncoder()
        elif args['encoder'] == 'twoconv':
            self.encoder = TwoConvEncoder()
        else:
            AssertionError("Not Implement!")

        self.Loss = nn.CrossEntropyLoss()

    def to_device(self, x):
        return torch.from_numpy(x).to(self.device)

    def grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def forward_feature(self, x):
        pass

    @abstractmethod
    def forward_cls(self, feature, y=None, classifier=None):
        pass

    @abstractmethod
    def finetune_classifier(self, num_classes=10):
        pass