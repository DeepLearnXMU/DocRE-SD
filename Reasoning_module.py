import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention,BertSelfOutput
from transformers import BertPreTrainedModel
from torch.nn import Softmax
import math
import copy
import pdb
from transformers.models.bert.modeling_bert import ACT2FN


class Reasoning_layer(nn.Module):
    def __init__(self, config, num_layers=1):
        super().__init__()
        layer_list = []
        for i in range(num_layers):
            ccnet = CrissCrossAttention_layer66(config, 4)
            layer_list.append(ccnet)
        self.Layers = nn.ModuleList(layer_list)


    def forward(self,entity_pair_matrix, entity_pair_masks):
        b, n_e, _, d = entity_pair_matrix.size()
        node_index = (entity_pair_masks > 0)

        for i, layer_module in enumerate(self.Layers):
            entity_pair_matrix = layer_module(i,entity_pair_matrix, entity_pair_masks)  # [b,97,d]
            if(i<1):
                hidden = entity_pair_matrix

        return entity_pair_matrix,hidden


class Reasoning_module(BertPreTrainedModel):
    def __init__(self, config,num_layers=1,return_intermediate=True):
        super().__init__(config)
        self.config = copy.deepcopy(config)
        self.return_intermediate = return_intermediate
        self.hidden_size = self.config.hidden_size

        self.relation_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.Online_inference = Reasoning_layer(self.config,num_layers)
     
        self.beta = 0.99
        self._get_target_encoder()

    def set_requires_grad(self,model, val):
        for p in model.parameters():
            p.requires_grad = val

    def _get_target_encoder(self):
        self.target_inference = copy.deepcopy(self.Online_inference)
        self.set_requires_grad(self.target_inference, False)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(self):
        for current_params, ma_params in zip(self.Online_inference.parameters(), self.target_inference.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def The_third_Path(self,entity_pair_matrix,entity_pair_masks):
        with torch.no_grad():
            hidden1,_ = self.target_inference(entity_pair_matrix, entity_pair_masks)
            hidden1.detach_()
            logits1 = self.relation_layer(hidden1)
            logits1.detach_()

        return logits1,hidden1

    def forward(self,entity_pair_matrix, entity_pair_masks):
        b, n_e, _, d = entity_pair_matrix.size()
        node_index = (entity_pair_masks > 0.5)

        hidden1,hidden2 = self.Online_inference(entity_pair_matrix,entity_pair_masks)
        logits1 = self.relation_layer(hidden1)
        logits2 = self.relation_layer(hidden2)

        return logits1,logits2,hidden1,hidden2



# h->*->t
def logit1(hidden_states_new, node_indx, attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    col_states1 = hidden_states_new.unsqueeze(2).expand(b, n, n, n, d)  # h->*
    row_states = hidden_states_new.permute(0, 2, 1, 3).contiguous()
    row_states1 = row_states.unsqueeze(1).expand(b, n, n, n, d)  #  *->t
    # [L,n,2d]
    key_states = torch.cat((col_states1[node_indx], row_states1[node_indx]), dim=-1)  # h->*->t

    if attention_mask is not None:
        col_mask1 = attention_mask.unsqueeze(2).expand(b, n, n, n)
        row_mask = attention_mask.permute(0, 2, 1).contiguous()
        row_mask1 = row_mask.unsqueeze(1).expand(b, n, n, n)

        attention_mask = col_mask1[node_indx] * row_mask1[node_indx]  # [L,n]

        attention_mask = (1.0 - attention_mask) * -10000.0  # mask pad
        attention_mask = attention_mask.unsqueeze(1)  # [L,1,n]

    return key_states, attention_mask

# h->* t->*
def logit2(hidden_states_new, node_indx, attention_mask=None):
    b, n, _, d = hidden_states_new.size()

    col_states1 = hidden_states_new.unsqueeze(2).expand(b, n, n, n, d)  #  h->*
    col_states2 = hidden_states_new.unsqueeze(1).expand(b, n, n, n, d)  # t->*
    # [L,n,2d]
    key_states = torch.cat((col_states1[node_indx], col_states2[node_indx]), dim=-1)  # h->* t->*
    if attention_mask is not None:
        col_mask1 = attention_mask.unsqueeze(2).expand(b, n, n, n)
        col_mask2 = attention_mask.unsqueeze(1).expand(b, n, n, n)

        attention_mask = col_mask1[node_indx] * col_mask2[node_indx]  # [L,n]

        attention_mask = (1.0 - attention_mask) * -10000.0  # mask pad
        attention_mask = attention_mask.unsqueeze(1)  # [L,1,n]

    return key_states, attention_mask

# h<-*<-t
def logit3(hidden_states_new, node_indx, attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    col_states2 = hidden_states_new.unsqueeze(1).expand(b, n, n, n, d)  # t->*
    row_states = hidden_states_new.permute(0, 2, 1, 3).contiguous()
    row_states2 = row_states.unsqueeze(2).expand(b, n, n, n, d)  #    *->h
    # [L,n,2d]
    key_states = torch.cat((col_states2[node_indx], row_states2[node_indx]), dim=-1)  # t->*->h

    if attention_mask is not None:
        col_mask2 = attention_mask.unsqueeze(1).expand(b, n, n, n)
        row_mask = attention_mask.permute(0, 2, 1).contiguous()
        row_mask2 = row_mask.unsqueeze(2).expand(b, n, n, n)

        attention_mask = col_mask2[node_indx] * row_mask2[node_indx]  # [L,n]

        attention_mask = (1.0 - attention_mask) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)  # [L,1,n]

    return key_states, attention_mask

# h<-*->t
def logit4(hidden_states_new, node_indx, attention_mask=None):
    b, n, _, d = hidden_states_new.size()
    row_states = hidden_states_new.permute(0, 2, 1, 3).contiguous()
    row_states1 = row_states.unsqueeze(1).expand(b, n, n, n, d)  #    *->t
    row_states2 = row_states.unsqueeze(2).expand(b, n, n, n, d)  #    *->h
    # [L,n,2d]
    key_states = torch.cat((row_states2[node_indx], row_states1[node_indx]), dim=-1)  # *->h *->t

    if attention_mask is not None:
        row_mask = attention_mask.permute(0, 2, 1).contiguous()
        row_mask1 = row_mask.unsqueeze(1).expand(b, n, n, n)
        row_mask2 = row_mask.unsqueeze(2).expand(b, n, n, n)

        attention_mask = row_mask2[node_indx] * row_mask1[node_indx]  # [L,n]
        attention_mask = (1.0 - attention_mask) * -10000.0  # mask pad
        attention_mask = attention_mask.unsqueeze(1)  # [L,1,n]

    return key_states, attention_mask

class Highway(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)

        self.H = nn.Linear(config.hidden_size, config.hidden_size)
        self.T = nn.Linear(config.hidden_size, config.hidden_size)
        self.T.bias.data.fill_(-1.)
        self.intermediate_act_fn = ACT2FN["gelu"]

        self.dense3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states,input_tensor):
        hidden_states = self.dropout1(self.dense1(hidden_states))
        h = self.intermediate_act_fn(self.H(hidden_states))
        t = torch.sigmoid(self.T(hidden_states))
        c = 1. - t
        hidden_states2 = h * t + hidden_states * c

        hidden_states = self.dropout3(self.dense3(hidden_states2))
        hidden_states = self.LayerNorm3(hidden_states + input_tensor)

        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)

        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size*2)
        self.intermediate_act_fn = ACT2FN["gelu"]
        # self.intermediate_act_fn = nn.ReLU(inplace=True)

        self.dense3 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.LayerNorm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states1 = self.LayerNorm1(hidden_states + input_tensor)

        hidden_states = self.dense2(hidden_states1)
        hidden_states2 = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense3(hidden_states2)
        hidden_states = self.dropout3(hidden_states)
        hidden_states = self.LayerNorm3(hidden_states + hidden_states1)
        return hidden_states

class CrissCrossAttention_layer66(nn.Module):
    def __init__(self, config, num_attention_heads):
        super().__init__()
        self.config = config
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.bin_k = nn.ModuleList(
            [nn.Linear(self.attention_head_size * 2, self.attention_head_size) for i in range(num_attention_heads)])
        self.bin_v = nn.ModuleList(
            [nn.Linear(self.attention_head_size * 2, self.attention_head_size) for i in range(num_attention_heads)])

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.Output = Highway(self.config)

        self.logit_function = {
            0: logit1,
            1: logit2,
            2: logit3,
            3: logit4,
        }

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, x.size(-1) // self.num_attention_heads)
        x = x.view(*new_x_shape)  # [b,n,h,d]

        if (len(list(x.size())) == 3):
            return x
        elif (len(list(x.size())) == 4):
            return x.permute(0, 2, 1, 3)
        else:
            return x.permute(0, 3, 1, 2, 4)

    def forward(self, layer_idx, hidden_states, attention_mask, label_01=None):
        node_indx = (attention_mask > 0.5)  # [b,n,n]
        new_hidden_states = hidden_states[node_indx]

        key_layer = self.transpose_for_scores(self.key(new_hidden_states))  # [L,h,d]
        value_layer = self.transpose_for_scores(self.value(new_hidden_states))  # [L,h,d]
        query_layer = self.transpose_for_scores(self.query(new_hidden_states))  # [L,h,d]

        new_hidden_states = torch.zeros(hidden_states.size()[:-1] + (key_layer.size(-1),)).to(key_layer)  # [L,n,n,d]
        # [L,n,2d]  [L,1,n]
        key_layers, value_layers, attention_masks = [], [], []
        for i in range(self.num_attention_heads):
            j = i % 4
            new_hidden_states[node_indx] = key_layer[:, i]  # [L,d]->[b,n_e,n_e,d]
            key_state, attention_mask1 = self.logit_function[j](new_hidden_states, node_indx, attention_mask)

            new_hidden_states[node_indx] = value_layer[:, i]  # [L,d]
            value_state, _ = self.logit_function[j](new_hidden_states, node_indx)

            key_state = self.bin_k[i](key_state)
            value_state = self.bin_v[i](value_state)

            key_state = torch.cat((key_layer[:, i].unsqueeze(1), key_state), dim=1)  # [L,n+1,d]
            value_state = torch.cat((value_layer[:, i].unsqueeze(1), value_state), dim=1)  # [L,n+1,d]
            mask = torch.zeros((attention_mask1.size(0), 1, 1)).to(attention_mask1)  # [L,1,1]
            attention_mask1 = torch.cat((mask, attention_mask1), dim=-1)  # [L,1,n+1]

            key_layers.append(key_state)
            value_layers.append(value_state)
            attention_masks.append(attention_mask1)

        key_layer = torch.stack(key_layers, dim=1)  # [L,h,n+1,d]
        value_layer = torch.stack(value_layers, dim=1)  # [L,h,n+1,d]
        attention_mask = torch.cat(attention_masks, dim=1)  # [L,h,n+1]

        query_layer = query_layer.unsqueeze(2)  # [L,h,1,d]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [L,h,1,n+1]
        attention_scores = attention_scores / math.sqrt(key_layer.size(-1))

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(2)  # [L,h,1,n+1]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)  # [L,h,d]
        new_context_layer_shape = context_layer.size()[:-2] + (context_layer.size(-1) * context_layer.size(-2),)
        context_layer = context_layer.view(*new_context_layer_shape)  # [L,d]

        Out_put = torch.zeros_like(hidden_states)
        Out_put[node_indx] = self.Output(context_layer, hidden_states[node_indx])

        return Out_put