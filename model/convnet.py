import torch
import torch.nn as nn
import torch.nn.functional as F
from model.template import Template

class ConvNet(Template):
    def __init__(self, args):
        super(ConvNet, self).__init__(args)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.args['num_classes'])
        )

    def forward_feature(self, x):
        x = x.unsqueeze(1).float()
        feature = self.encoder(x)
        return feature

    def forward_classifier(self, feature, y=None, classifier=None):
        if classifier is None:
            preds = self.classifier(feature.mean(-1).mean(-1))
        else:
            preds = classifier(feature.mean(-1).mean(-1))

        if y is not None:
            y = y.long()
            probs = nn.Softmax(dim=-1)(preds)
            entropy = torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            loss = self.Loss(preds, y) - entropy
            return probs, loss
        return preds

    def forward(self, x, y=None):
        feature = self.forward_feature(x)
        return self.forward_classifier(feature, y)

    def finetune_classifier(self, num_classes=10):
        classifier = nn.Sequential(
            nn.Linear(self.encoder.out_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        return classifier.cuda()