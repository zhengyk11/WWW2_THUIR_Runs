# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2019 THUIR. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()

INF = 1e30

class DotAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(DotAttention, self).__init__()
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.linear_inputs = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_memory = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_res = nn.Linear(input_size * 2, hidden_size, bias=False)
        self.linear_gate = nn.Linear(hidden_size, hidden_size, bias=False)

    def softmax_mask(self, val, mask):
        return val + INF * (mask - 1)

    def forward(self, inputs, memory, mask, batch_size, inputs_len, memory_len):
        inputs_relu = self.relu(self.linear_inputs(self.dropout(inputs)))
        memory_relu = self.relu(self.linear_memory(self.dropout(memory)))
        outputs = torch.bmm(inputs_relu, torch.transpose(memory_relu, 1, 2)) / (self.hidden_size ** 0.5)
        logits = self.softmax_mask(outputs, mask).view(batch_size*inputs_len, memory_len)
        logits_softmax = self.softmax(logits).view(batch_size, inputs_len, memory_len)
        outputs_memory = torch.bmm(logits_softmax, memory)
        res = self.linear_res(torch.cat((inputs, outputs_memory), dim=2))
        gate = self.sigmoid(self.linear_gate(self.dropout(res)))
        return res * gate

class Summ(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(Summ, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.linear_memory = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_s0 = nn.Linear(hidden_size, 1, bias=False)
        self.linear_res = nn.Linear(input_size, hidden_size, bias=False)

    def softmax_mask(self, val, mask):
        return val + INF * (mask - 1)

    def forward(self, memory, mask, batch_size, memory_len):
        s0 = self.tanh(self.linear_memory(self.dropout(memory)))
        s = self.linear_s0(self.dropout(s0))
        s1 = self.softmax_mask(s.view(batch_size, memory_len), mask.view(batch_size, memory_len))
        a = self.softmax(s1).view(batch_size, memory_len, 1)
        res = self.linear_res(self.dropout((a * memory).sum(dim=1)))
        return res

class PointerNet(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(PointerNet, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, dropout=dropout_rate)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)

        self.softmax = nn.Softmax(dim=-1)
        self.pointer_1 = pointer(hidden_size, dropout_rate)
        self.pointer_2 = pointer(hidden_size, dropout_rate)

    def forward(self, init, match, mask, batch_size, max_para_len):
        d_match = self.dropout(match)
        init_mask = Variable(torch.ones(batch_size, self.hidden_size))
        if use_cuda:
            init_mask = init_mask.cuda()
        init_mask_drop = self.dropout(init_mask)
        inp, logits1 = self.pointer_1(d_match, init*init_mask_drop, mask, batch_size, max_para_len)
        d_inp = self.dropout(inp)
        self.gru.flatten_parameters()
        _, state = self.gru(d_inp.view(batch_size, 1, self.hidden_size), init.view(1, batch_size, self.hidden_size))
        state = state.view(batch_size, self.hidden_size)
        _, logits2 = self.pointer_2(d_match, state*init_mask_drop, mask, batch_size, max_para_len)
        logits1 = self.softmax(logits1)
        logits2 = self.softmax(logits2)
        return logits1, logits2

class pointer(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(pointer, self).__init__()
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.linear_u = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.linear_s0 = nn.Linear(hidden_size, 1, bias=False)

    def softmax_mask(self, val, mask):
        return val + INF * (mask - 1)

    def forward(self, inputs, state, mask, batch_size, max_para_len):
        state = state.view(batch_size, 1, self.hidden_size)
        state_expand = state.expand(batch_size, max_para_len, self.hidden_size)
        u = torch.cat((state_expand, inputs), dim=2)
        s0 = self.tanh(self.linear_u(self.dropout(u)))
        s = self.linear_s0(self.dropout(s0))
        s1 = self.softmax_mask(s.view(batch_size, -1), mask.view(batch_size, -1))
        a = self.softmax(s1).view(batch_size, -1, 1)
        res = (a * inputs).sum(dim=1)
        return res, s1
