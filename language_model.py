import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from  fc import FCNet
import torch.nn.functional as F

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size, hidden_dim, num_layers=1):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embed.weight)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers)
        # init LSTM
        self.init_lstm(self.lstm.weight_ih_l0)
        self.init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

    def init_lstm(self, weight):
        # init LSTM in chunks of 4 cells
        for w in weight.chunk(4, 0):
            nn.init.xavier_uniform_(w)
    def forward(self, q_ind, seq_length):
        embedding = self.embed(q_ind)

        embedding = nn.utils.rnn.pack_padded_sequence(embedding, seq_length.to("cpu"), batch_first=True)
        _, h = self.lstm(embedding)
        return h[0][0]  # return final hidden state of LSTM

class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='LSTM'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output

class QuestionSelfAttention(nn.Module):
    def __init__(self, num_hid, dropout):
        super(QuestionSelfAttention, self).__init__()
        self.num_hid = num_hid
        self.drop = nn.Dropout(dropout)
        self.W1_self_att_q = FCNet([num_hid, num_hid])
        self.W2_self_att_q = FCNet([num_hid, 1])

    def forward(self, ques_feat):
        '''
        ques_feat: [batch, 14, num_hid]
        '''
        batch_size = ques_feat.shape[0]
        q_len = ques_feat.shape[1]

        # (batch*14,num_hid)
        ques_feat_reshape = ques_feat.contiguous().view(-1, self.num_hid)
        # (batch, 14)
        atten_1 = self.W1_self_att_q(ques_feat_reshape)
        atten_1 = torch.tanh(atten_1)
        atten = self.W2_self_att_q(atten_1).view(batch_size, q_len)
        # (batch, 1, 14)
        weight = F.softmax(atten.t(), dim=1).view(-1, 1, q_len)
        ques_feat_self_att = torch.bmm(weight, ques_feat)
        ques_feat_self_att = ques_feat_self_att.view(-1, self.num_hid)
        # (batch, num_hid)
        ques_feat_self_att = self.drop(ques_feat_self_att)
        return ques_feat_self_att