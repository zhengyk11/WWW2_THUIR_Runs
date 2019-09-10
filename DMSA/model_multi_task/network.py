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
import logging
from layers import DotAttention, Summ, PointerNet

use_cuda = torch.cuda.is_available()
INF = 1e30

class MultiTaskNetwork(nn.Module):
    def __init__(self, args, vocab_size, pretrain_embedding=None):
        super(MultiTaskNetwork, self).__init__()
        self.args = args
        self.logger = logging.getLogger("ntcir14")

        self.hidden_size = args.hidden_size
        self.embed_size = args.embed_size
        self.dropout_rate = args.dropout_rate

        self.bm25_predictor = Network(args, vocab_size, pretrain_embedding)
        self.cm_predictor = Network(args, vocab_size, pretrain_embedding)

        self.question_gru = nn.GRU(self.embed_size * 2, self.hidden_size, bidirectional=True,
                                  batch_first=True, dropout=self.dropout_rate)
        self.passage_gru = nn.GRU(self.hidden_size * 4, self.hidden_size, bidirectional=True,
                                     batch_first=True, dropout=self.dropout_rate)

        self.summ_question = Summ(self.hidden_size * 2, self.hidden_size, self.dropout_rate)
        self.summ_passage = Summ(self.hidden_size * 2, self.hidden_size, self.dropout_rate)
        self.linear_score_1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear_score_2 = nn.Linear(self.hidden_size + 3, self.hidden_size)
        self.linear_score_3 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, question, passage, bm25_target_score):

        batch_size, max_question_len = question.size()
        max_passage_len = passage.size()[1]

        question_mask = (question.data.cpu().numpy() > 0).astype(np.float32)
        question_mask = Variable(torch.from_numpy(question_mask)).view(batch_size, 1, max_question_len)
        if use_cuda:
            question_mask = question_mask.cuda()

        passage_mask = (passage.data.cpu().numpy() > 0).astype(np.float32)
        passage_mask = Variable(torch.from_numpy(passage_mask)).view(batch_size, 1, max_passage_len)
        if use_cuda:
            passage_mask = passage_mask.cuda()

        bm25_score, bm25_passage_rep, bm25_question_rep = self.bm25_predictor(question, passage)
        cm_score, cm_passage_rep, cm_question_rep = self.cm_predictor(question, passage)

        question_encode = torch.cat((bm25_question_rep, cm_question_rep), dim=2)
        # print 'question_encode', question_encode.size()
        init_gru_state = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if use_cuda:
            init_gru_state = init_gru_state.cuda()
        question_context, _ = self.question_gru(question_encode, init_gru_state)

        passage_encode = torch.cat((bm25_passage_rep, cm_passage_rep), dim=2)
        # print 'passage_encode', passage_encode.size()
        init_gru_state = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if use_cuda:
            init_gru_state = init_gru_state.cuda()
        passage_context, _ = self.passage_gru(passage_encode, init_gru_state)

        question_state = self.summ_question(question_context, question_mask, batch_size, max_question_len)
        # print 'question_state', question_state.size()
        passage_state = self.summ_passage(passage_context, passage_mask, batch_size, max_passage_len)
        # print 'passage_state', passage_state.size()
        question_passage_state = torch.cat((question_state, passage_state), dim=1)
        score_hidden = self.linear_score_2(torch.cat((
                                    self.sigmoid(self.linear_score_1(self.dropout(question_passage_state))),
                                    bm25_score, cm_score, bm25_target_score.view(-1, 1)), dim=1))
        score = self.linear_score_3(self.sigmoid(self.dropout(score_hidden))).contiguous()
        return score, bm25_score.contiguous(), cm_score.contiguous()


class Network(nn.Module):
    def __init__(self, args, vocab_size, pretrain_embedding=None):
        super(Network, self).__init__()
        self.args = args
        self.logger = logging.getLogger("ntcir14")
        self.embed_size = args.embed_size   # 300 as default
        self.hidden_size = args.hidden_size # 150 as default
        self.vocab_size = vocab_size
        self.dropout_rate = args.dropout_rate
        self.encode_gru_num_layer = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        if pretrain_embedding is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(pretrain_embedding))
            self.logger.info('model load pretrained embedding successfully.')

        self.question_encode_gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True,
                                          batch_first=True, dropout=self.dropout_rate, num_layers=self.encode_gru_num_layer)
        self.passage_encode_gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True,
                                         batch_first=True, dropout=self.dropout_rate, num_layers=self.encode_gru_num_layer)
        self.early_match_gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True,
                                           batch_first=True, dropout=self.dropout_rate)
        self.late_match_gru = nn.GRU(self.hidden_size * 4, self.hidden_size, bidirectional=True,
                                      batch_first=True, dropout=self.dropout_rate)
        self.early_late_gru = nn.GRU(self.hidden_size * 2, self.hidden_size, bidirectional=True,
                                            batch_first=True, dropout=self.dropout_rate)
        self.question_passage_gru = nn.GRU(self.hidden_size * 4, self.hidden_size, bidirectional=True,
                                           batch_first=True, dropout=self.dropout_rate)

        # self.dot_attention_question = DotAttention(self.hidden_size * 2, self.hidden_size, self.dropout_rate)

        self.dot_attention_late_passage = DotAttention(self.embed_size, self.hidden_size, self.dropout_rate)
        self.dot_attention_late_question = DotAttention(self.embed_size, self.hidden_size, self.dropout_rate)
        self.dot_attention_late_match = DotAttention(self.hidden_size * 2, self.hidden_size, self.dropout_rate)

        self.dot_attention_early_match = DotAttention(self.embed_size, self.hidden_size, self.dropout_rate)
        self.dot_attention_early_passage = DotAttention(self.hidden_size * 2, self.hidden_size, self.dropout_rate)

        self.summ_question = Summ(self.embed_size, self.hidden_size, self.dropout_rate)
        self.summ_passage = Summ(self.hidden_size * 2, self.hidden_size, self.dropout_rate)
        self.linear_score = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, question, passage):

        batch_size, max_question_len = question.size()
        max_passage_len = passage.size()[1]

        # embed stage
        question_embed = self.embedding(question)  # batch_size, max_ques_len, embed_size
        passage_embed = self.embedding(passage)  # batch_size, max_para_len, embed_size

        # early matching stage
        question_mask = (question.data.cpu().numpy() > 0).astype(np.float32)
        question_mask = Variable(torch.from_numpy(question_mask)).view(batch_size, 1, max_question_len)
        if use_cuda:
            question_mask = question_mask.cuda()
        question_mask_expand = question_mask.expand(batch_size, max_passage_len, max_question_len).contiguous()
        early_match = self.dot_attention_early_match(passage_embed, question_embed,
                                                     question_mask_expand,
                                                     batch_size, max_passage_len, max_question_len)
        # print 'early_match', early_match.size()

        init_gru_state = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if use_cuda:
            init_gru_state = init_gru_state.cuda()
        early_match_context, _ = self.early_match_gru(early_match, init_gru_state)

        # print 'early_match_context', early_match_context.size()
        # early passage self-attention stage
        passage_mask = (passage.data.cpu().numpy() > 0).astype(np.float32)
        passage_mask = Variable(torch.from_numpy(passage_mask)).view(batch_size, 1, max_passage_len)
        if use_cuda:
            passage_mask = passage_mask.cuda()
        early_match_att = self.dot_attention_early_passage(early_match_context, early_match_context, passage_mask,
                                                                   batch_size, max_passage_len, max_passage_len)
        # print 'early_match_att', early_match_att.size()

        # late question self-attention stage
        late_match_question_att = self.dot_attention_late_question(question_embed, question_embed, question_mask,
                                                             batch_size, max_question_len, max_question_len)
        # print 'late_match_question_att', late_match_question_att.size()
        # late passage self-attention stage
        late_match_passage_att = self.dot_attention_late_passage(passage_embed, passage_embed, passage_mask,
                                                             batch_size, max_passage_len, max_passage_len)

        init_gru_state = Variable(torch.zeros(self.encode_gru_num_layer*2, batch_size, self.hidden_size))
        if use_cuda:
            init_gru_state = init_gru_state.cuda()
        question_encode_embed, _ = self.question_encode_gru(late_match_question_att, init_gru_state)
        init_gru_state = Variable(torch.zeros(self.encode_gru_num_layer*2, batch_size, self.hidden_size))
        if use_cuda:
            init_gru_state = init_gru_state.cuda()
        passage_encode_embed, _ = self.passage_encode_gru(late_match_passage_att, init_gru_state)
        # print 'question_encode_embed', question_encode_embed.size()
        # print 'passage_encode_embed', passage_encode_embed.size()

        late_match_att = self.dot_attention_late_match(passage_encode_embed, question_encode_embed,
                                             question_mask_expand,
                                             batch_size, max_passage_len, max_question_len)
        # print 'late_match_att', late_match_att.size()

        early_late_encode = torch.cat((early_match_att, late_match_att), dim=2)
        # print 'early_late_encode', early_late_encode.size()
        init_gru_state = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if use_cuda:
            init_gru_state = init_gru_state.cuda()
        early_late_context, _ = self.early_late_gru(early_late_encode, init_gru_state)
        # print 'early_late_context', early_late_context.size()

        # prediction stage
        question_state = self.summ_question(question_embed, question_mask, batch_size, max_question_len)
        # print 'question_state', question_state.size()
        passage_state = self.summ_passage(early_late_context, passage_mask, batch_size, max_passage_len)
        # print 'passage_state', passage_state.size()
        question_passage_state = torch.cat((question_state, passage_state), dim=1)
        score = self.linear_score(question_passage_state)
        # print 'score', score.size()
        return score, early_late_context, question_embed

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    class config():
        def __init__(self):
            self.embed_size = 200
            self.hidden_size = 100
            self.dropout_rate = 0.2

    args = config()
    model = MultiTaskNetwork(args, 10)
    q = Variable(torch.zeros(8, 40).long())
    p = Variable(torch.zeros(8, 100).long())
    bm25 = Variable(torch.zeros(8, 1).float())
    model(q, p, bm25)
    print count_parameters(model)
