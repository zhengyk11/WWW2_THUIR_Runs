import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

import torch.nn.functional as F
from model3 import *
import scipy.stats as stats
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def Tensor2Varible(tensor_):
    var = Variable(tensor_)
    var = var.cuda() if use_cuda else var
    return var

class BaselineReader(nn.Module):
    def __init__(self, vocab_size, config):
        super(BaselineReader, self).__init__()
        self.vocab_size = vocab_size
        self.embsize = config['embsize']
        self.term_hidden_size = config['term_hidden_size']
        self.query_hidden_size = config['query_hidden_size']
        self.sentence_hidden_size = config['sentence_hidden_size']
        self.evidence_hidden_size = config['evidence_hidden_size']
        self.position_hidden_size = config['position_hidden_size']

        self.size_filter = config['size_filter']
        self.n_filter = config['n_filter']
        self.n_repeat = config['n_repeat']
        self.drate = config['drate']
        self.mask_id = config['mask_id']
        self.prate = config['prate']
        self.pairwise = config['pairwise']
        self.baseline_type = config['baseline_type']

        self.vertical_decay = config['vertical_decay']

        self.num_selector_class = config['num_selector_class']
        self.finish_size = self.sentence_hidden_size*2 + self.position_hidden_size  + self.evidence_hidden_size
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embsize)

        pre_word_embeds_addr = config['emb'] if 'emb' in config else None

        if pre_word_embeds_addr is not None:
            print 'Loading word embeddings'
            pre_word_embeds = cPickle.load(open(pre_word_embeds_addr))
            print 'pre_word_embeds size: ',pre_word_embeds.shape
            self.embedding_layer.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))

        self.position_embedding = nn.Embedding(10, self.position_hidden_size)  # set fixed number

        self.term_encoder = nn.GRU(self.embsize, self.term_hidden_size)
        self.cnn = KimCNN(self.size_filter, self.n_filter, self.embsize)

        #self.input_cnn_sim = ConcatCnnNet(self.size_filter, self.n_filter,self.sentence_hidden_size)
        #self.input_cnn_xor = ConcatCnnNet(self.size_filter, self.n_filter, self.sentence_hidden_size)

        self.input_cnn_sim = CnnNet(self.sentence_hidden_size)
        self.input_cnn_xor = CnnNet(self.sentence_hidden_size)

        self.selector = Selector(self.query_hidden_size,self.sentence_hidden_size,\
                                 self.evidence_hidden_size,num_class=self.num_selector_class)
        self.pos_rnn = EvidenceRNN(self.sentence_hidden_size*2,self.evidence_hidden_size)
        #self.neg_rnn = EvidenceRNN(self.sentence_hidden_size, self.evidence_hidden_size)

        #if self.baseline_type == 'independent':
        self.dense = nn.Linear(self.sentence_hidden_size*2,self.evidence_hidden_size)
        self.finish_net = FinishNet(self.finish_size)
        self.score = nn.Linear(self.evidence_hidden_size * 10, 1)
        self.drop = nn.Dropout(self.drate)

        self.weight_s = nn.Linear(self.embsize, 1)

    def get_mask(self, input_q, input_d):
        query_mask = 1 - torch.eq(input_q, 0).float()
        sent_mask = 1 - torch.eq(input_d, 0).float()
        input_mask = torch.bmm(query_mask.unsqueeze(-1), sent_mask.unsqueeze(1))
        return input_mask

    def query_document_match(self, query_batch, sent_batch):
        '''
        :param query_batch:
        :param query_lengths:
        :param sent_batch:
        :param sent_lengths:
        :return:
        '''
        # input_q = Tensor2Varible(torch.LongTensor(query_batch))
        query_vectors = self.embedding_layer(query_batch)  # (batch_size,max_len,embsize)

        # input_d = Tensor2Varible(torch.LongTensor(query_batch))
        sent_vectors = self.embedding_layer(sent_batch)  # #(batch_size,l,embsize)

        #print query_vectors
        q_norm = query_vectors.norm(p=2, dim=-1, keepdim=True)
        d_norm = sent_vectors.norm(p=2, dim=-1, keepdim=True)

        norm_q_embed = query_vectors / q_norm
        norm_d_embed = sent_vectors / d_norm

        sim_matrix = torch.bmm(norm_q_embed, norm_d_embed.transpose(1, 2))  # (b_s,q,d)
        input_mask = self.get_mask(query_batch, sent_batch)
        exact_matrix = torch.gt(sim_matrix, 0.99).float()

        exact_matrix = exact_matrix * input_mask
        sim_matrix = sim_matrix * input_mask

        batch_size = query_vectors.size(0)
        #print 'query_vectors: ',query_vectors.size()
        x = self.weight_s(query_vectors).squeeze(-1)  # ((b_s,q)
        y = self.weight_s(sent_vectors).squeeze(-1)  # (b_s,k)

        xx = x.view(batch_size, -1, 1).expand_as(sim_matrix)
        yy = y.view(batch_size, 1, -1).expand_as(sim_matrix)

        S_cos = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), sim_matrix.unsqueeze(-1)], dim=-1)  # (b_s,q,k,3)
        S_xor = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), exact_matrix.unsqueeze(-1)], dim=-1)  # (b_s,q,k,3)

        q_d_vectors_sim = self.input_cnn_sim(S_cos.transpose(1,3))
        q_d_vectors_xor = self.input_cnn_xor(S_xor.transpose(1,3))

        q_d_vectors = torch.cat([q_d_vectors_sim,q_d_vectors_xor],dim=-1)
        return q_d_vectors  # (b_s,sent_hidden_size * 2)

    def kmax_pooling(self,x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self,query_batch,query_lengths,doc_batch,doc_lengths,gt_rels=None):
        '''
        :param query_variable: (b_s,n_word) padded
        :param document_variable:(b_s,n_sent,n_word) padded
        :param gt_rels: (b_s,1) float
        :return:
        '''

        query_var = Tensor2Varible(torch.LongTensor(query_batch))
        doc_batch = Tensor2Varible(torch.LongTensor(doc_batch))

        if not self.pairwise:
            targets = Tensor2Varible(torch.FloatTensor(gt_rels))
            targets = targets.view(-1,1)

        self.b_s = doc_batch.size(0)
        self.n_sent = doc_batch.size(1)  # l
        self.n_word = doc_batch.size(2)  # k
        sent_hidden = self.pos_rnn.initHidden(self.b_s)

        output_matrix = Tensor2Varible(torch.zeros(self.b_s, self.n_sent, self.evidence_hidden_size))

        for sid in xrange(self.n_sent):
            sent_input = doc_batch[:,sid,:]
            sent_mask = 1 - torch.eq(sent_input, 0).float()
            q_sent_input = self.query_document_match(query_var,sent_input)
            sent_mask = torch.gt(torch.sum(sent_mask, 1), 0).view(-1, 1).float()


            if self.baseline_type == 'independent':
                output_matrix[:, sid, :] = self.dense(q_sent_input) * sent_mask

            else:
                outputs, pos_hidden_new = self.pos_rnn(q_sent_input.unsqueeze(0), sent_hidden)
                sent_hidden = sent_hidden * (1.0 - sent_mask) + pos_hidden_new * sent_mask

                outputs = self.drop(outputs)

                if self.vertical_decay:
                    fit_alpha = 4.368;
                    fit_loc = -1.359;
                    fit_beta = 1.366;
                    weight_output = stats.gamma.pdf(sid + 1, a=fit_alpha, scale=fit_beta, loc=fit_loc)
                    outputs = outputs * (weight_output + 0.6)

                outputs = outputs * sent_mask
                output_matrix[:, sid, :] = outputs.squeeze(0)

        output = self.kmax_pooling(output_matrix,dim=1,k=10).view(self.b_s,-1)

        if self.pairwise:
            rel_predicted = nn.Tanh()(self.score(output))  # b_s * 1
            #rel_predicted = nn.Sigmoid()(self.score(output))  # b_s * 1
            mse = None
        else:
            rel_predicted = nn.Sigmoid()(self.score(output))#b_s * 1
            mse = (rel_predicted - targets) ** 2

        rl_cost = mse

        return rel_predicted,rl_cost



    def position_emb(self, word_index, max_length, split=10):
        unit = (max_length + 1) / split
        p_index = word_index / unit

        p_index[max_length < split] = 0
        p_index = np.clip(p_index,0,split-1)

        position_input = Variable(torch.LongTensor([p_index]))[0]
        position_input = position_input.cuda() if use_cuda else position_input
        pos_emb = self.position_embedding(position_input).unsqueeze(0)

        return pos_emb


















