import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.template import Template

class ModFC(nn.Module):
    def __init__(self, in_features: int, out_features: int, context_features: int, bias: bool = True,
                 activation=None, eps: float = 1e-6):
        super(ModFC, self).__init__()
        self.native_weight = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(out_features, in_features), a=math.sqrt(5)
            )
        )
        if bias:
            self.native_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.native_bias = None
        self.scale = nn.Linear(context_features, in_features, bias=False)
        self.activation = activation
        self.eps = eps

    def forward(self, input, context):
        """

        :param input: (b, in_num, in_dim)
        :param context:  (1, out_num, out_dim)
        :return:
        """
        if input.size(1) != context.size(1):
            input = input.unsqueeze(2)
            context = context.unsqueeze(1)

        scale = F.layer_norm(self.scale(context), [input.shape[-1]])
        output = F.linear(input * scale, self.native_weight, bias=self.native_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

class DynamicRouting(nn.Module):
    def __init__(self, in_attr_dim, out_attr_num, out_attr_dim, routing_iter, dropout_p, modulated: bool = True):
        super(DynamicRouting, self).__init__()

        self.in_attr_dim = in_attr_dim
        self.out_attr_num = out_attr_num
        self.out_attr_dim = out_attr_dim
        self.routing_iter = routing_iter # number of iteration T (start from 0)

        self.modulated = modulated
        if self.modulated:
            self.modfc = ModFC(self.in_attr_dim, self.out_attr_dim, self.out_attr_dim)
            self.codes = nn.Parameter(
                nn.init.kaiming_uniform_(
                    torch.empty(1, self.out_attr_num, self.out_attr_dim), a=math.sqrt(5)
                )
            )
        else:
            self.w = nn.Linear(self.in_attr_dim, self.out_attr_num * self.out_attr_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.transform = nn.Sequential(
            nn.Linear(self.in_attr_dim, self.out_attr_dim),
        )

    def routing_once(self, u_pred, u, v, b):
        """
        u_pred: (b, in_attr, out_attr, out_dim)
        u: (b, in_attr, 1, out_dim)
        v: (b, out_attr, out_dim)
        b: (b, in_attr, out_attr)
        """
        batch_size, in_attr, out_attr, out_dim = u_pred.size()

        v_sqz = v.unsqueeze(1) # (b, 1, out_attr, out_dim)
        b = b + F.cosine_similarity(u, v_sqz, dim=-1)
        c = F.softmax(b, dim=-1).view(-1, in_attr, out_attr, 1)
        c = self.dropout(c)

        s = (c * u_pred).sum(dim=1) / (1e-8 + c.sum(1))
        return s, c, b

    def routing_kmeans(self, u, u_pred, return_c=False):
        """
        :param u: (b, in_attr, out_dim)
        :param u_pred: (b, in_attr, out_attr, out_dim)
        :return:
        """
        batch_size, in_attr, out_attr, out_dim = u_pred.size()  # (b, in_attr, out_attr, out_dim)
        device = u_pred.device

        # first iteration
        v = u_pred.mean(1)
        b = torch.zeros((in_attr, out_attr)).to(device)
        b = b.expand((batch_size, in_attr, out_attr))  # (b, in_attr, out_attr)
        s = v

        c = torch.ones_like(b).unsqueeze(-1).cuda() / self.out_attr_num
        for i in range(1, self.routing_iter + 1):
            s, c, b = self.routing_once(u_pred, u.unsqueeze(2), s, b)

        if self.routing_iter >= 1:
            v = v + s

        if return_c:
            return v, c
        else:
            return v

    def primary_attrs(self, x):
        B, C, H, W = x.size()
        pri_attrs = x.view(B, -1, self.in_attr_dim, H, W)
        pri_attrs = pri_attrs.permute(0, 1, 3, 4, 2).contiguous()
        pri_attrs = pri_attrs.view(pri_attrs.size(0), -1, pri_attrs.size(4))  # (B, in_attr, in_dim)
        return pri_attrs

    def forward(self, feature, add_module=False):
        u = self.primary_attrs(feature)  # (b, in_attr, in_dim)

        if self.modulated:
            if add_module:
                codes = torch.cat([self.codes, self.finetune_codes], dim=1)
            else:
                codes = self.codes
            u_pred = self.modfc(u, codes)  # (b, in_attr, out_attr, out_dim)
        else:
            u_pred = self.w(u)  # (b, in_attr, out_attr*out_dim)
            u_pred = u_pred.view(feature.size(0), -1, self.out_attr_num, self.out_attr_dim)  # (b, in_attr, out_attr*out_dim)

        u = self.transform(u)
        v, c = self.routing_kmeans(u, u_pred, return_c=True)
        v = F.relu(v)
        return v, c

    def add_modules(self, add_num: int = 0):
        if add_num > 0:
            self.finetune_codes = nn.Parameter(
                nn.init.kaiming_uniform_(torch.empty(1, add_num, self.out_attr_dim, device='cuda'), a=math.sqrt(5)),
                requires_grad=True).cuda()

class SMN(Template):
    def __init__(self, args):
        super(SMN, self).__init__(args)
        in_channels = self.encoder.out_channels

        self.out_attr_num = 2
        self.out_attr_dim = 384
        print("out_attr_num:", self.out_attr_num, " out_attr_dim:", self.out_attr_dim)
        self.routing_module = DynamicRouting(in_channels, self.out_attr_num, self.out_attr_dim, args['routing_iter'], args['dropout'])

        self.modulated = args['modulated']
        self.classifier = nn.Linear(self.out_attr_dim, args['num_classes'])
        self.loss_coef = args['im_coef']

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward_feature(self, x):
        x = x.unsqueeze(1).float()
        features = self.encoder(x)
        return features

    def forward_classifier(self, features, y=None, classifier=None, add_module=False):
        out, c = self.routing_module(features, add_module)
        importance = c.squeeze(-1).sum(0).sum(0)
        importance_loss = self.cv_squared(importance)
        importance_loss *= self.loss_coef

        if classifier is None:
            classifier = self.classifier
        preds = classifier(out.mean(1))

        if y is not None:
            y = y.long()
            probs = nn.Softmax(dim=-1)(preds)
            entropy = torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            loss = self.Loss(preds, y) - entropy
            if add_module:
                return probs, loss
            else:
                return probs, loss + importance_loss
        return preds

    def forward(self, x, y=None):
        features = self.forward_feature(x)
        return self.forward_classifier(features, y)

    def finetune_classifier(self, num_classes=10):
        classifier = nn.Linear(self.out_attr_dim, num_classes).cuda()
        return classifier

    def get_finetune_params(self, classifier, add_modules=False):
        params = []
        for param in classifier.parameters():
            params.append(param)

        if add_modules:
            params.append(self.routing_module.finetune_codes)
        return params