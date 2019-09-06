import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn import functional, init
from torch.nn import Parameter
import sys,cPickle
use_cuda = torch.cuda.is_available()


def repackage_var(vs, requires_grad = False):
    print 'vs:',vs
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(vs) == Variable:
        return Variable(vs.data, requires_grad = requires_grad)
    elif type(vs) == Parameter:
        return Parameter(vs.data,requires_grad = requires_grad)
    else:
        return tuple(repackage_var(v) for v in vs)

class CnnNet(torch.nn.Module):
    def __init__(self,config):
        super(CnnNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, 1, 1), #1, 3, 3, 1, 1
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1), #3, 5, 3, 1, 1
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Linear(640, config['sentence_hidden_size']) #5 * 2 * 10

    def forward(self, x):
        x = x.unsqueeze(1)  # (N,Ci,len,embsize)
        #print 'x: ',x.size()
        conv1_out = self.conv1(x)
        #print 'conv1_out: ',conv1_out.size()
        conv2_out = self.conv2(conv1_out)
        #print 'conv2_out: ',conv2_out.size()
        res = conv2_out.view(conv2_out.size(0), -1)
        out = self.dense(res)
        return out

class KimCNN(nn.Module):
    '''
    Encode sentence vector
    '''
    def __init__(self, size_filter, n_out_kernel, embsize, drate=0.01):
        super(KimCNN, self).__init__()
        self.size_filter = size_filter
        self.drate = drate
        self.n_filter = n_out_kernel
        Ci = 1  # n_in_kernel
        Co = self.n_filter  # args.kernel_num
        Ks = self.size_filter #range(1,self.size_filter+1)[1, 2, 3, 4, 5]
        self.n_out = len(Ks) * self.n_filter
        #self.embedding = nn.Embedding(vocab_size, embsize)

        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embsize)) for K in Ks])


    def forward(self, x):
        '''
        x: (b_s, len, embsize)
        '''
        #x = self.embedding(x)
        x = x.unsqueeze(1)  # (N,Ci,len,embsize)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,len), ...]*len(Ks)
        self.fea_map = x1
        x2 = [nn.MaxPool1d(i.size(2))(i).squeeze(2) for i in x1]  # [(N,Co), ...]*len(Ks)
        x3 = torch.cat(x2, 1)  # (b_s, co*len(Ks))
        return x3


class Selector(nn.Module):
    def __init__(self,query_size,sent_size,hidden_size,num_class=3):
        '''
        :param query_size: query embedding size
        :param sent_size: sentence embedding size
        :param hidden_size: RNN hidden size
        '''
        super(Selector, self).__init__()
        #self.policy_function = nn.Linear(sent_size + hidden_size, 3)
        self.Wp = nn.Linear(sent_size + hidden_size, num_class)

        #torch.nn.init.xavier_uniform_(self.Wp.weight)

        # hc: [query, sent]
        self.Whc = nn.Linear(query_size + sent_size, sent_size)
        #print 'query_size + sent_size:',query_size, sent_size
        #self.v = nn.Parameter(torch.FloatTensor(1, sent_size))


    def forward(self,q_sent_input,hidden):
        '''
        :param query_input:(b_s,querySize)
        :param sent_input:(b_s,sentSize)
        :param hidden:(1,b_s,hiddenSize)
        :return:
        '''
        '''
        hc = torch.cat([query_input, sent_input], dim=-1)
        alpha = F.tanh(self.Whc(hc))#(b_s,senSize)
        #att = F.log_softmax(torch.mul(self.v,alpha), dim=1)#(b_s,senSize)
        att = F.log_softmax(alpha, dim=1)  # (b_s,senSize)
        #att = alpha
        #print 'att:',alpha[:2,:10]
        q_sen_input = torch.mul(att,sent_input)#(b_s,senSize)
        '''

        state = torch.cat([q_sent_input,hidden.squeeze(0)],-1)


        scaled_out = nn.Sigmoid()(self.Wp(state))
        #scaled_out = torch.abs(out)
        #print 'scaled_out:', out,scaled_out

        scaled_out = torch.clamp(scaled_out,min=1e-5, max=1 - 1e-5)
        #scaled_out[torch.isnan(scaled_out)] = 1e-5

        scaled_out = F.normalize(scaled_out, p=1, dim=1)
        #print 'scaled_out:',scaled_out
        #s = torch.sum(self.policy_function.weight.data)
        #print 'weight: ',s
        #(b_s,senSize),(b_s,3)
        return scaled_out



class FinishNet(nn.Module):
    def __init__(self, input_size):
        super(FinishNet, self).__init__()
        self.W_finish = nn.Linear(input_size,2)
        #torch.nn.init.xavier_uniform_(self.W_finish.weight)

    def forward(self,sent_input,position_emb,pos_emb):
        #print 's1: ',sent_input.size()
        #print 's2: ', position_emb.size()
        #print 's3: ', pos_emb.size()
        #print 's4: ', neg_emb.size()
        state = torch.cat([sent_input,pos_emb,position_emb],-1)
        #print state.size()
        scaled_out = nn.Sigmoid()(self.W_finish(state))
        scaled_out = F.normalize(scaled_out, p=1, dim=-1)

        #print 'scaled_out:',scaled_out
        scaled_out = torch.clamp(scaled_out, 1e-5, 1 - 1e-5)
        return scaled_out


class EvidenceRNN(nn.Module):
    def __init__(self,input_size, hidden_size, n_layers=1):
        super(EvidenceRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        output = input

        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self,batch_size=1):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        else:
            return hidden



###########################################################
def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()

def save(model, filename):
    say('saving model to {} ...\n'.format(filename))
    params_value = [value.get_value() for value in model.params]
    with open(filename, 'w') as f:
        cPickle.dump(params_value, f, -1)


def load(model, filename):
    say('load model from {} ...\n'.format(filename))
    if filename.endswith('.pkl'):
        with open(filename, 'r') as f:
            params_value = cPickle.load(f)
        assert len(params_value) == len(model.params)
        for i in xrange(len(model.params)):
            model.params[i].set_value(params_value[i])
    else:
        raise NotImplementedError