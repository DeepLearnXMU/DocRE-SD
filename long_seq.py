import torch
import torch.nn.functional as F
import numpy as np
import pdb

def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True
    )
    sequence_output = output['last_hidden_state']
    attention = (output["attentions"][-1] + output["attentions"][-2] + output["attentions"][-3]) / 3.0
    
    hidden_states = output["hidden_states"][7:]
    return sequence_output, attention, hidden_states

def process_long_input1(model, input_ids, attention_mask, start_tokens, end_tokens):
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    Pad = torch.tensor([0]).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512: 
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        sequence_output = output['last_hidden_state']
        attention = (output["attentions"][-1]+output["attentions"][-2]+output["attentions"][-3])/3.0
        
        hidden_states=output["hidden_states"][7:]
    else:
        new_input_ids,new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist() 
        for i, l_i in enumerate(seq_len): 
            if l_i <= 512:  
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else: 
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)  
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1) 
                attention_mask1 = attention_mask[i, :512]   
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]   

                new_input_ids.extend([input_ids1, input_ids2])

                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)

        input_ids = torch.stack(new_input_ids, dim=0)

        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        sequence_output = output['last_hidden_state']  
        attention = (output["attentions"][-1]+output["attentions"][-2]+output["attentions"][-3])/3.0    
        
        hidden_states = output["hidden_states"][7:] 

        len_h=len(hidden_states)  
        i = 0   
        new_output, new_attention, new_hidden_states = [], [],[]
        for (n_s, l_i) in zip(num_seg, seq_len):  
            if n_s == 1:    
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512)) 
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)

                Temp_hidden_states=[]
                for k in range(len_h):
                    h_states = F.pad(hidden_states[k][i], (0, 0, 0, c - 512))  
                    Temp_hidden_states.append(h_states)     
                new_hidden_states.append(Temp_hidden_states)    
            elif n_s == 2:  
                
                output1 = sequence_output[i][:512 - len_end]  
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))  
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                Temp_hidden_states1 = []
                for k in range(len_h):
                    h_states = hidden_states[k][i][:512 - len_end]  
                    h_states = F.pad(h_states, (0, 0, 0, c - 512 + len_end))  
                    Temp_hidden_states1.append(h_states)  
                
                output2 = sequence_output[i + 1][len_start:]    
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])

                Temp_hidden_states2 = []
                for k in range(len_h):
                    h_states = hidden_states[k][i + 1][len_start:]  
                    h_states = F.pad(h_states, (0, 0, l_i - 512 + len_start, c - l_i))  
                    Temp_hidden_states2.append(h_states)  

                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)

                Temp_hidden_states = []
                for k in range(len_h):
                    h_states = Temp_hidden_states1[k]+Temp_hidden_states2[k]/ mask.unsqueeze(-1)
                    Temp_hidden_states.append(h_states)  

                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
                new_hidden_states.append(Temp_hidden_states)  
            i += n_s

        hidden_states=[]
        batch_size=len(new_hidden_states)
        for k in range(len_h):
            hidden_states_temp=[]
            for b in range(batch_size):
                hidden_states_temp.append(new_hidden_states[b][k])
            hidden_states_temp = torch.stack(hidden_states_temp, dim=0)  
            hidden_states.append(hidden_states_temp)    

        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)

    return sequence_output, attention,hidden_states
