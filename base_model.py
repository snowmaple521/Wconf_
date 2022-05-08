import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np
def calc_entropy(p):
    # size(att) = [b x g x v x q]
    eps = 1e-8
    return (-p * (p + eps).log()).sum(1)
def mask_softmax(x,mask):
    #x: [512,36,1] = att
    mask=mask.unsqueeze(2).float() #[512,36] all:0
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon #512,1,1
    x4=x3/x3_sum.expand_as(x3)
    return x4


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier,att_q_a,att_v_a,q_net2):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.q_emb2 = q_net2
        self.v_net = v_net
        self.classifier = classifier
        self.att_q_a = att_q_a
        self.att_v_a = att_v_a
        self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)

    def forward(self, v, q, labels, bias,v_mask,epoch):
        """Forward

        v: [batch, num_objs, obj_dim] [512,36,2048]
        b: [batch, num_objs, b_dim][512,2274]
        q: [batch_size, seq_length][512,14]
        v_mask:1:None 2:0

        return: logits, not probs
        """
        # v = v.float()
        q = q.long()
        w_emb = self.w_emb(q) #torch.Size([512, 14, 300])
        q_emb = self.q_emb(w_emb)  # torch.Size([512, 1024]) [batch, q_dim]
        # print('q', q.shape)
        # print('w_emb', w_emb.shape)
        # print('q_emb',q_emb.shape)
        att = self.v_att(v, q_emb) #[512,36,1]
        if v_mask is None:
            att = nn.functional.softmax(att, 1) #If v_mask=None ,only use att softmax
        else:
            att= mask_softmax(att,v_mask) # If v_mask not None , use mask_softmax
        #after add attention v_feature
        v_emb = (att * v).sum(1)  # [512,36,1]*[512,36,2048]=  [512,2048] [batch, v_dim]

        q_repr = self.q_net(q_emb) #[512,1024]
        v_repr = self.v_net(v_emb) #[512,1024]

        joint_repr = q_repr * v_repr #[512,1024]

        logits = self.classifier(joint_repr) #[512,2274]

        if labels is not None:
            loss = self.debias_loss_fn(joint_repr, logits, bias, labels)
        else:
            loss = None
        # if qnoty is not None:
        #     w_emb_type = self.w_emb(qnoty)
        #     return logits, loss, w_emb_type
        # else:
        return logits, loss, w_emb

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    v_att1 = NewAttention(dataset.v_dim, dataset.num_ans_candidates, num_hid)
    q_net = FCNet([num_hid, num_hid])
    q_net2 = FCNet([300, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier,v_att1)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(2048, q_emb.num_hid, num_hid) #v_att + q
    q_att_a = NewAttention(300, dataset.num_ans_candidates, num_hid)
    v_att_a = NewAttention(2048, dataset.num_ans_candidates, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    q_net2 = FCNet([300, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier,q_att_a,v_att_a,q_net2)