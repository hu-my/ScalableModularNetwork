# Truncated Kernel
# See "Dynamic inference with neural interpreters"
# https://arxiv.org/abs/2110.06399
#
# Author: Nasim Rahaman
#
# The code is based on the Pytorch implementation:
# https://github.com/nasimrahaman/neural-interpreter/blob/main/neural_interpreters/core/kernels.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.template import Template
from model.smn import ModFC
from typing import Optional
from contextlib import contextmanager


def get_kernel(name):
    return globals()[name]


class Kernel(nn.Module):
    TRUNCATION_IMPLEMENTATIONS = {"direct", "masked"}

    def __init__(
        self,
        truncation: Optional[float] = None,
        straight_through: bool = True,
        initial_bandwidth: float = 1.0,
        learnable_bandwidth: bool = True,
        truncation_implementation: str = "direct",
    ):
        super(Kernel, self).__init__()
        # Validate
        assert truncation_implementation in self.TRUNCATION_IMPLEMENTATIONS, (
            f"Unknown implementation: {truncation_implementation}; "
            f"it should be one of: {self.TRUNCATION_IMPLEMENTATIONS}."
        )
        # Privates
        self._return_distance = False
        self._return_log_kernel = False
        # Publics
        self.truncation = truncation
        self.straight_through = straight_through
        self.learnable_bandwidth = learnable_bandwidth
        self.truncation_implementation = truncation_implementation
        if self.learnable_bandwidth:
            self.bandwidth = nn.Parameter(
                torch.tensor(float(initial_bandwidth), dtype=torch.float)
            )
        else:
            self.register_buffer(
                "bandwidth", torch.tensor(float(initial_bandwidth), dtype=torch.float)
            )

    @contextmanager
    def return_distance(self):
        old_return_distance = self._return_distance
        self._return_distance = True
        yield
        self._return_distance = old_return_distance

    @contextmanager
    def return_log_kernel(self):
        old_return_log_kernel = self._return_log_kernel
        self._return_log_kernel = True
        yield
        self._return_log_kernel = old_return_log_kernel

    @property
    def returning_distance(self):
        return self._return_distance

    @property
    def returning_log_kernel(self):
        return self._return_log_kernel

    def truncated_gaussian(self, distance: torch.Tensor) -> torch.Tensor:
        # Early exit if we're returning distance
        if self.returning_distance:
            return distance
        kwargs = dict(
            distance=distance,
            bandwidth=self.bandwidth,
            truncation=self.truncation,
            straight_through=self.straight_through,
            returning_log_kernel=self.returning_log_kernel
        )
        if self.truncation_implementation == "direct":
            kernel = self._direct_truncate(**kwargs)
        else:
            raise NotImplementedError
        return kernel

    @staticmethod
    def _direct_truncate(
        distance: torch.Tensor,
        bandwidth: torch.Tensor,
        truncation: float,
        straight_through: bool = True,
        returning_log_kernel: bool = False
    ) -> torch.Tensor:
        # Scale distance by bandwidth right away for stable gradients
        distance = pre_truncation_distance = distance / bandwidth
        # Truncate distances if required. All distances above a threshold are
        # yeeted to inf.
        if truncation is not None:
            truncated_distance = torch.where(
                distance > (truncation / bandwidth.detach()),
                torch.empty_like(distance).fill_(float("inf")),
                distance,
            )
            if straight_through:
                distance = distance + (truncated_distance - distance).detach()
            else:
                distance = truncated_distance
        if returning_log_kernel:
            # In this case, we're returning the log of what would otherwise be the
            # output of this class.
            kernel = -distance
        else:
            # Exponentiate. This will have the min possible distance (= 0) always
            # mapped to 1.
            kernel = (-distance).exp()
        return kernel

    def forward(self, signatures: torch.Tensor, types: torch.Tensor):
        # Function signature: UC, BVC -> BUV
        # signatures.shape = UC
        # types.shape = BVC
        raise NotImplementedError


class DotProductKernel(Kernel):
    def normalize(self, signatures: torch.Tensor, types: torch.Tensor):
        # signatures.shape = ...C
        # types.shape = ...C
        signatures = F.normalize(signatures, p=2, dim=-1)
        types = F.normalize(types, p=2, dim=-1)
        return signatures, types

    def forward(
        self,
        signatures: torch.Tensor,
        types: torch.Tensor,
        einsum_program: Optional[str] = None,
    ):
        # Function signature: {BUC,UC}, BVC -> BUV
        # signatures.shape = UC or BUC
        # types.shape = BVC
        # First normalize both variables before computing inner product
        signatures, types = self.normalize(signatures, types)
        # Compute the dot product distance,
        # defined as (1 - cosine_similarity)
        if einsum_program is not None:
            pass
        elif signatures.dim() == 2:
            einsum_program = "uc,bvc->bvu"
        else:
            assert signatures.dim() == 3
            einsum_program = "buc,bvc->bvu"
        # distance.shape = BUV
        distance = 1.0 - torch.einsum(einsum_program, signatures, types)
        # Compute and return the kernel
        kernel = self.truncated_gaussian(distance)
        return kernel

class SMN_kernel(Template):
    def __init__(self, args):
        super(SMN_kernel, self).__init__(args)
        self.in_channels = self.encoder.out_channels

        self.k = 1
        self.num_experts = 2
        self.hidden_size = 384
        self.output_size = 384
        print("k:{}, number of experts:{}".format(self.k, self.num_experts))

        # self.experts = nn.ModuleList(
        #    [MLP(in_channels, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.modfc = ModFC(self.in_channels, self.output_size, self.hidden_size)
        self.codes = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(1, self.num_experts, self.output_size), a=math.sqrt(5)
            )
        )
        self.signatures = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(self.num_experts, self.output_size), a=math.sqrt(5)
            )
        )
        self.kernel = DotProductKernel()
        self.transform = nn.Linear(self.in_channels, self.output_size)
        self.classifier = nn.Linear(self.output_size, args['num_classes'])

    def forward_feature(self, x):
        x = x.unsqueeze(1).float()
        features = self.encoder(x)
        return features

    def primary_attrs(self, x):
        B, C, H, W = x.size()
        pri_attrs = x.view(B, -1, self.in_channels, H, W)
        pri_attrs = pri_attrs.permute(0, 1, 3, 4, 2).contiguous()
        pri_attrs = pri_attrs.view(pri_attrs.size(0), -1, pri_attrs.size(4))  # (B, in_attr, in_dim)
        return pri_attrs

    def forward_classifier(self, features, y=None, classifier=None):
        # out, _ = self.routing_module(features)
        features = self.primary_attrs(features)
        function_variable_affinities = self.kernel(self.signatures, self.transform(features))
        out = self.modfc(features, self.codes)
        out = out * function_variable_affinities.unsqueeze(-1)

        if classifier is None:
            classifier = self.classifier

        preds = classifier(out.mean(1).mean(1))

        if y is not None:
            y = y.long()
            probs = nn.Softmax(dim=-1)(preds)
            entropy = torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            loss = self.Loss(preds, y) - entropy
            return probs, loss
        return preds

    def forward(self, x, y=None):
        features = self.forward_feature(x)
        return self.forward_classifier(features, y)

    def finetune_classifier(self, num_classes=10):
        classifier = nn.Linear(self.output_size, num_classes).cuda()
        # classifier.weight.data[:, :dim] = self.classifier.weight.data.clone().detach()
        # classifier.bias.data[:dim] = self.classifier.bias.data.clone().detach()
        return classifier

    def get_finetune_params(self, classifier):
        params = []
        for param in classifier.parameters():
            params.append(param)

        return params

