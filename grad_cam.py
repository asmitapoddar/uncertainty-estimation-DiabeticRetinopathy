#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.autograd import Variable, Function

class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.to(self.device)

    def _encode_multilabel(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        for i in range(idx):
            one_hot[0][i] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        # self.probs = F.softmax(self.preds, dim=1)[0]
        self.probs = F.sigmoid(self.preds)
        # self.prob, self.idx = self.probs.sort(0, True)
        # return self.prob, self.idx
        return self.probs[0]

    def backward(self, idx):
        multi_label= torch.FloatTensor(np.expand_dims(idx,axis=0)).to(self.device)
        multi_one_hot = torch.stack((1-multi_label, multi_label),1)
        # ml = self._encode_multilabel(idx)
        # one_hot = self._encode_one_hot(idx)
        # ml = self._encode_multilabel(idx)
        self.multi_probs = torch.stack((1-self.probs, self.probs), 1)
        self.multi_probs.backward(gradient=multi_one_hot, retain_graph=True)


class BackPropagation(_PropagationBase):
    def generate(self):
        output = self.image.grad.detach().cpu().numpy()
        return output.transpose(0, 2, 3, 1)[0]


class GuidedBackPropagation(BackPropagation):
    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


class Deconvolution(BackPropagation):
    def __init__(self, model):
        super(Deconvolution, self).__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)

class IntegratedGradient(_PropagationBase):
    def __init__(self, model, steps):
        super(IntegratedGradient, self).__init__(model)
        self.steps = steps

    def generate(self, idx):
        grad = 0
        inp_data = self.image.data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            self.forward(new_inp)
            self.backward(idx)
            g = new_inp.grad.data
            grad += g

        output= grad * inp_data / self.steps
        return output.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
        

class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.detach().cpu().numpy()
