import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels,weight=None):
        
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        
        logit1 = logits - (1 - p_mask) * 1e30   
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        
        loss = loss1 + loss2

        if weight is not None:
            weight *= 2
            weight += 1
            loss = loss * weight

        loss = 10*loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)  
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)  
            top_v = top_v[:, -1]      
            mask = (logits >= top_v.unsqueeze(1)) & mask  
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)  
        return output

def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    p_loss = p_loss.sum(-1)
    q_loss = q_loss.sum(-1)
    loss = (p_loss + q_loss) / 2
    return loss.mean()*10.0


class Distillation_loss(nn.Module):
    def __init__(self,Temp=1.0):
        super().__init__()
        self.Temp = Temp
        self.Loss = nn.CrossEntropyLoss()

    def forward(self,S,T):
        '''
        :param inputs: [L,97]
        :param targets: [L,97]
        :return:
        '''
        T = F.softmax(T/self.Temp,dim=-1)
        loss = self.Loss(S/self.Temp,T)

        return loss