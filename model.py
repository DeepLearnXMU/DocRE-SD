import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss,compute_kl_loss,Distillation_loss
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
import math
import time
import copy
import pdb
import torch.nn.functional as F
from Reasoning_module import Reasoning_module
from BERT import BertModel
import ujson as json
import random

def contrast_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return (2 - 2 * (x * y).sum(dim=-1))/10.0

class DocREModel(nn.Module):
    def __init__(self, args,config,emb_size=768, block_size=64, num_labels=-1,used_cross_attnetion=False):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.num_labels=num_labels
        config.output_hidden_states = True 
        
        self.encoder = BertModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        self.loss_fnt = ATLoss()
        self.Dist_Loss = Distillation_loss(Temp=1)

        self.hidden_size = config.hidden_size
        self.CCNet_d=512

        self.ELU = nn.ELU()
        self.Tach=nn.Tanh()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.BCE1 = nn.BCEWithLogitsLoss(reduction='none')
        self.BCE = nn.BCEWithLogitsLoss()

        self.MASK_token = nn.parameter.Parameter(nn.init.xavier_uniform_(
            torch.empty(1, self.config.hidden_size)).float())

        self.head_pair = nn.Sequential(
            nn.Linear(config.hidden_size*3, emb_size),
            nn.Tanh(),
        )
        self.tail_pair = nn.Sequential(
            nn.Linear(config.hidden_size*3, emb_size),
            nn.Tanh(),
        )
        self.entity_pair_extractor = nn.Sequential(
            nn.Linear(emb_size * block_size, emb_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.Reasoning_module = Reasoning_module(self.config, num_layers=2)

        self.emb_size = emb_size
        self.block_size = block_size
        self.Step =0
        self.Total =0
        self.init_weights()

    def init_weights(self):
        for m in [self.head_pair,self.tail_pair,self.entity_pair_extractor,self.Reasoning_module]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    
    def Encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta": 
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        
        sequence_output, attention,hidden_states= process_long_input(self.encoder, input_ids, attention_mask, start_tokens, end_tokens)

        
        return sequence_output, attention,hidden_states
    
    def get_mention_rep(self, sequence_output, attention, attention_mask, entity_pos):
        mention = []
        entity2mention = []
        Entity_attention = []

        batch_size, doc_len, _ = sequence_output.size()
        Max_met_num = -1
        for i in range(batch_size):
            mention.append([sequence_output[i][0]])
            mention_indx = 1

            entity2mention.append([])
            entity_atts = []
            mention_atts = []

            for j, e in enumerate(entity_pos[i]):
                e_att = []
                entity2mention[-1].append([])
                for start, end, sentence_id in e:
                    mention[-1].append((sequence_output[i][start + 1] + sequence_output[i][end]) / 2.0)
                    e_att.append((attention[i, :, start + 1] + attention[i, :, end]) / 2.0)
                    entity2mention[-1][-1].append(mention_indx)
                    mention_indx += 1
                mention_atts.extend(e_att)

                if len(e_att) > 1:
                    e_att = torch.stack(e_att, dim=0).mean(0)
                else:
                    e_att = e_att[0]
                entity_atts.append(e_att)

            entity_atts = torch.stack(entity_atts, dim=0)
            Entity_attention.append(entity_atts)
            mention_atts = torch.stack(mention_atts, dim=0)

            if (Max_met_num < mention_indx):
                Max_met_num = mention_indx

        for i in range(batch_size):
            origin_len = len(mention[i])
            extence = Max_met_num - origin_len
            for j in range(extence):
                mention[i].append(torch.zeros(768).to(sequence_output.device))
            mention[i] = torch.stack(mention[i], dim=0)

        mention = torch.stack(mention, dim=0)

        mention_feature = {
            "mention": mention,
            "entity2mention": entity2mention,
            "Entity_attention": Entity_attention,
        }
        return mention_feature


    def Create_mask_matrix(self,entity_pair_matrix,entity_pair_masks,Labels):
        r_max = min(0.5,self.Step/self.Total+0.2)
        r = random.uniform(0.1, r_max)
        
        label = Labels[..., 0]  
        neg_MASK = label - F.dropout(label, p=r/2.0) * (1.0-r/2.0)  

        label1 = (1 - label) * entity_pair_masks
        pos_MASK = label1 - F.dropout(label1, p=r) * (1-r)  
        predict_position = ((pos_MASK + neg_MASK) > 0.5).float()
        
        hidden_states = entity_pair_matrix.clone()
        MASK_index = (predict_position > 0.5)
        hidden_states[MASK_index] = self.MASK_token

        return hidden_states,predict_position

    def Entity_level_predict( self,Entity_feature):
        entity_pair_matrix=Entity_feature["entity_pair_matrix"]
        entity_pair_masks=Entity_feature["entity_pair_masks"]
        hss=Entity_feature["hss"]
        tss=Entity_feature["tss"]
        hts_tensor=Entity_feature["hts_tensor"]
        rss=Entity_feature["rss"]
        label_01 = Entity_feature["label_01"]
        entitys = Entity_feature["entitys"]

        loss, loss1 = 0, 0
        Index = (entity_pair_masks > 0)

        if label_01 is not None:
            entity_pair_matrix1, predict_position = self.Create_mask_matrix(entity_pair_matrix, entity_pair_masks, label_01)
            logits0,logits01,hidden0,hidden01 = self.Reasoning_module(entity_pair_matrix, entity_pair_masks)
            logits1,logits11,hidden1,hidden11 = self.Reasoning_module(entity_pair_matrix1, entity_pair_masks)
            #An exponential moving average (EMA) branch to stabilize the training of the mask branch
            logits02,hidden02 = self.Reasoning_module.The_third_Path(entity_pair_matrix,entity_pair_masks)

            logits = logits0[Index]
            labels = label_01[Index]
            loss += self.loss_fnt(logits.float(), labels)

            logits = logits1[Index]
            labels = label_01[Index]
            loss += self.loss_fnt(logits.float(), labels)

            loss1 += compute_kl_loss(logits1[Index], logits0[Index])

            loss1 += self.Dist_Loss(logits1[Index], logits02[Index])

            Index_local = (predict_position > 0.5)
            loss1 += contrast_loss_fn(hidden1[Index_local], hidden02[Index_local]).mean()

            logit = logits0
        else:
            logit1 = self.Reasoning_module(entity_pair_matrix,entity_pair_masks)[0]
            label1 = logit1 * (Index.float()).unsqueeze(-1)  
            label1[Index] = self.loss_fnt.get_label(label1[Index], num_labels=self.num_labels)

            logit = logit1

        logit1 = []
        for i, ht in enumerate(hts_tensor):
            logit1.append(logit[i][ht[:, 0], ht[:, 1], :])
        logit1 = torch.cat(logit1, 0)  

        return logit1,loss,loss1

    
    def get_entity_pair(self, mention_feature, encoder_out, hts, encoder_output, encoder_mask, labels=None):
        sequence_output = mention_feature["mention"]
        Entity_attention = mention_feature["Entity_attention"]  
        entity2mention = mention_feature["entity2mention"]

        hss, tss, rss, Entity_pairs, hts_tensor, entitys = [], [], [], [], [],[]
        batch_size = sequence_output.size()[0]
        Pad = torch.zeros((1, self.config.hidden_size)).to(sequence_output.device)  
        for i in range(batch_size):
            entity_embs = []
            for j, e in enumerate(entity2mention[i]):
                e_index = torch.LongTensor(e).to(sequence_output.device)
                e_emb = torch.logsumexp(sequence_output[i].index_select(0, e_index), dim=0)
                entity_embs.append(e_emb)

            entity_embs = torch.stack(entity_embs, dim=0)       
            entitys.append(entity_embs)

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hts_tensor.append(ht_i)

            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            doc_rep = sequence_output[i][0][None, :].expand(hs.size()[0], 768)

            h_att = torch.index_select(Entity_attention[i], 0, ht_i[:, 0])  
            t_att = torch.index_select(Entity_attention[i], 0, ht_i[:, 1])

            ht_att = (h_att * t_att).sum(1)  
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", encoder_out[i], ht_att)  

            hs_pair = self.head_pair(torch.cat([hs, rs,doc_rep], dim=1))    
            ts_pair = self.tail_pair(torch.cat([ts, rs,doc_rep], dim=1))    
            
            b1 = hs_pair.view(-1, self.hidden_size // self.block_size, self.block_size)
            b2 = ts_pair.view(-1, self.hidden_size // self.block_size, self.block_size)
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.hidden_size* self.block_size)
            entity_pair = self.entity_pair_extractor(bl)  

            hss.append(hs)  
            tss.append(ts)  
            rss.append(rs)

            entity_pair = torch.cat([Pad, entity_pair], dim=0)  
            Entity_pairs.append(entity_pair)  

        Max_entity_num = max([len(x) for x in entity2mention])
        entity_pair_index = torch.zeros((batch_size, Max_entity_num, Max_entity_num)).long().to(sequence_output.device)
        label_01 = torch.zeros((batch_size, Max_entity_num, Max_entity_num, 97)).float()  

        for i, ht_i in enumerate(hts_tensor):
            index = torch.arange(ht_i.size()[0]).to(sequence_output.device) + 1  
            entity_pair_index[i][ht_i[:, 0], ht_i[:, 1]] = index
            if labels is not None:
                label = torch.tensor(labels[i]).float()  
                label_01[i][ht_i[:, 0], ht_i[:, 1]] = label

        entity_pair_masks = (entity_pair_index != 0).float()
        label_01 = label_01.to(sequence_output.device)

        entity_pair_matrix = []
        for i in range(batch_size):
            entity_pair_matrix.append(Entity_pairs[i][entity_pair_index[i]])
            pad_l = Max_entity_num - entitys[i].size(0)
            entitys[i] = torch.cat([entitys[i],] + [Pad] * pad_l, dim=0)

        entity_pair_matrix = torch.stack(entity_pair_matrix, dim=0)

        hss = torch.cat(hss, dim=0)  
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        entitys = torch.stack(entitys)

        if labels is None:
            label_01 = None

        Entity_feature = {
            "entity_pair_matrix": entity_pair_matrix,
            "entity_pair_masks": entity_pair_masks,
            "hss": hss,
            "tss": tss,
            "rss": rss,
            "hts_tensor": hts_tensor,
            "label_01": label_01,
            "entitys":entitys,
        }
        return Entity_feature

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ):
        sequence_output, attention,hidden_states = self.Encode(input_ids, attention_mask)
        sequence_output = (hidden_states[-1] + hidden_states[-2] + hidden_states[-3]) / 3.0

        mention_feature=self.get_mention_rep(sequence_output,attention,attention_mask,entity_pos)

        Entity_feature=self.get_entity_pair(
            mention_feature,
            sequence_output,
            hts,
            sequence_output,
            attention_mask,
            labels=labels
        )
        logits,loss,loss1 = self.Entity_level_predict(Entity_feature)
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)  

        if labels is not None: 
            output = ((loss1+loss)/2.0,loss1) + output

        return output 
