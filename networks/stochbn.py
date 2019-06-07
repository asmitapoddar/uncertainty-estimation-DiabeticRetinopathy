import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.autograd import Variable


def mean_features(x):
    out = x.mean(0)
    while out.dim() > 1:
        out = out.mean(1)
    return out


class _MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, mode=None,
                 learn_stats=False, batch_size=0):
        super(_MyBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.stats_momentum = momentum
        self.strategy = 'batch'

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.register_buffer('cur_mean', torch.zeros(num_features))
        self.register_buffer('cur_var', torch.ones(num_features))

        self.register_buffer('running_m', torch.zeros(num_features))
        self.register_buffer('running_m2', torch.zeros(num_features))
        self.register_buffer('running_logvar', torch.zeros(num_features))
        self.register_buffer('running_logvar2', torch.zeros(num_features))

        self.register_buffer('running_mean_mean', torch.zeros(num_features))
        self.register_buffer('running_mean_var', torch.ones(num_features))
        self.register_buffer('running_logvar_mean', torch.zeros(num_features))
        self.register_buffer('running_logvar_var', torch.ones(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def update_smoothed_stats(self):
        eps = 1e-6
        self.running_m = (1 - self.stats_momentum) * self.running_m + self.stats_momentum * self.cur_mean
        self.running_m2 = (1 - self.stats_momentum) * self.running_m2 + self.stats_momentum * (self.cur_mean ** 2)

        self.running_logvar = (1 - self.stats_momentum) * self.running_logvar + self.stats_momentum * torch.log(self.cur_var + eps)
        self.running_logvar2 = (1 - self.stats_momentum) * self.running_logvar2 + self.stats_momentum * (torch.log(self.cur_var + eps) ** 2)

        self.running_mean_mean.copy_(self.running_m)
        self.running_mean_var.copy_(self.running_m2 - (self.running_m ** 2))

        self.running_logvar_mean.copy_(self.running_logvar)
        self.running_logvar_var.copy_(self.running_logvar2 - (self.running_logvar ** 2))

        self.running_mean.copy_(self.running_m)
        self.running_var.copy_(torch.exp(self.running_logvar))

    def forward_stochbn(self, input):
        cur_mean = mean_features(input)
        cur_var = F.relu(mean_features(input**2) - cur_mean**2)

        self.cur_var.copy_(cur_var.data)
        self.cur_mean.copy_(cur_mean.data)

        running_mean_mean = Variable(self.running_mean_mean)
        running_mean_var = Variable(self.running_mean_var)

        running_logvar_mean = Variable(self.running_logvar_mean)
        running_logvar_var = Variable(self.running_logvar_var)

        if self.strategy == 'sample':
            eps = Variable(torch.randn(self.num_features))
            if self.weight.data.is_cuda:
                eps = eps.cuda()
            sampled_var = torch.exp(eps * torch.sqrt(running_logvar_var) + running_logvar_mean)
            vars = sampled_var
        elif self.strategy == 'running':
            vars = Variable(self.running_var)
        elif self.strategy == 'batch':
            vars = cur_var
        else:
            raise NotImplementedError('Unknown strategy: {}'.format(self.var_strategy))

        if self.strategy == 'sample':
            eps = Variable(torch.randn(self.num_features))
            if self.weight.data.is_cuda:
                eps = eps.cuda()
            sampled_mean = eps * torch.sqrt(running_mean_var) + running_mean_mean
            means = sampled_mean
        elif self.strategy == 'running':
            means = Variable(self.running_mean)
        elif self.strategy == 'batch':
            means = cur_mean
        else:
            raise NotImplementedError('Unknown strategy: {}'.format(self.mean_strategy))

        if self.training:
            self.update_smoothed_stats()

        return self.batch_norm(input, means, vars + self.eps)

    def forward(self, input):
        self._check_input_dim(input)

        return self.forward_stochbn(input)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class MyBatchNorm1d(_MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm1d, self)._check_input_dim(input)

    def batch_norm(self, input, means, vars):
        # TODO: implement for any dimensionality
        out = input - means.view(-1, self.num_features)
        out = out / torch.sqrt(vars.view(-1, self.num_features))
        out = out * self.weight.view(1, self.num_features)
        out = out + self.bias.view(1, self.num_features)
        return out


class MyBatchNorm2d(_MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm2d, self)._check_input_dim(input)

    def batch_norm(self, input, means, vars):
        # TODO: implement for any dimensionality
        if means.dim() == 1:
            means = means.view(-1, self.num_features, 1, 1)
            vars = vars.view(-1, self.num_features, 1, 1)
        out = input - means
        out = out / torch.sqrt(vars)
        out = out * self.weight.view(1, self.num_features, 1, 1)
        out = out + self.bias.view(1, self.num_features, 1, 1)
        return out
