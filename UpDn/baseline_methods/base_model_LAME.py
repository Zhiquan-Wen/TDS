import torch
import torch.nn as nn
from torch.utils import data
from attention import Attention, NewAttention, NewAttention1
from language_model import QuestionEmbedding2, WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, mask, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb, mask)  # 512, 36, 1
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits, joint_repr

# class BaseModel1(nn.Module):
#     def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
#         super(BaseModel, self).__init__()
#         self.w_emb = w_emb
#         self.q_emb = q_emb
#         self.v_att = v_att
#         self.q_net = q_net
#         self.v_net = v_net
#         self.classifier = classifier

#     def forward(self, v, b, q, mask, q_len, labels):
#         """Forward

#         v: [batch, num_objs, obj_dim]
#         b: [batch, num_objs, b_dim]
#         q: [batch_size, seq_length]

#         return: logits, not probs
#         """
#         w_emb = self.w_emb(q)
#         q_emb = self.q_emb(w_emb, q_len) # [batch, q_dim]

#         att = self.v_att(v, q_emb, mask)
#         v_emb = (att * v).sum(1) # [batch, v_dim]

#         q_repr = self.q_net(q_emb)
#         v_repr = self.v_net(v_emb)
#         joint_repr = q_repr * v_repr
#         logits = self.classifier(joint_repr)
#         return logits

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

# def build_baseline0_newatt_GQA(dataset, num_hid):
#     w_emb = WordEmbedding(dataset.vocab_dict.num_vocab, 300, 0.0)
#     q_emb = QuestionEmbedding2(300, num_hid, 1, False, 0.0)
#     v_att = NewAttention1(dataset.v_dim, q_emb.num_hid, num_hid)
#     q_net = FCNet([q_emb.num_hid, num_hid])
#     v_net = FCNet([dataset.v_dim, num_hid])
#     classifier = SimpleClassifier(
#         num_hid, num_hid * 2, dataset.answer_dict.num_vocab, 0.5)
#     return BaseModel1(w_emb, q_emb, v_att, q_net, v_net, classifier)