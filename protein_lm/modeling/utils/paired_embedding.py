import torch

def create_mask(indices = None):
    pass

def create_paired_position_ids(input_ids, sentinel_indices,pad_id=0):
    """
    a helper function to create aboslute position encodings for paired sequences
    e.g Assume condition sequence has length 3 and focus sequence has length 2
    <cond_seq></cond_seq><focus_seq></focus_seq> -> [0,1,2,3,0,1,2] instead of the normal [0,1,2,3,4,5,6]
    args:
    input_ids (torch.tensor): the input_ids vector passed in model_pytorch
    sentinel_indices (torch.tensor): the sentinel_indices vector passed in model_pytorch
    pad_id (int): value of the padding token
    returns:
    triplet of torch.tensor,torch.tensor,torch.tensor
    """
    print('input_ids.shape:',input_ids.shape)
    print('sentinel_indices:',sentinel_indices)
    batch_size = input_ids.size(0)
    n = input_ids.size(1)
    
    # Initialize the batch tensor with zeros
    
    
    # batch_tensor_focus = torch.zeros_like(input_ids)
    # Initialize the batch tensor with zeros
    # cond_idx,idx,focus_idx,end_idx = sentinel_indices[0].item()
    idx = sentinel_indices[0][1].item() # get index of end of condition sequence plus condition padding
    
    
    position_ids = torch.zeros_like(input_ids,device=input_ids.device)
    position_ids[:, :idx] = torch.arange(start = pad_id + 1, end=pad_id+1+idx, dtype=input_ids.dtype, device=input_ids.device)
    position_ids[:, idx:] = torch.arange(start = pad_id + 1, end=pad_id+1+n-idx, dtype=input_ids.dtype, device=input_ids.device)
    

    cond_pad_start_idx = sentinel_indices[:,0]
    cond_pad_end_idx = sentinel_indices[:,1] # this is the same as idx but a list of batch size
    cond_mask = torch.arange(position_ids.size(1),device=input_ids.device).expand(len(cond_pad_start_idx), position_ids.size(1)) >= cond_pad_start_idx.unsqueeze(1)
    cond_start_mask = cond_mask <= cond_pad_start_idx.unsqueeze(1)
    cond_end_mask = cond_mask >= cond_pad_start_idx.unsqueeze(1)
    cond_final_mask = cond_start_mask & cond_end_mask 
    print(f'cond_start_mask:{cond_start_mask}')
    print(f'cond_end_mask:{cond_end_mask}')
    print(f'cond_final_mask:{cond_final_mask}')

    focus_pad_start_idx = sentinel_indices[:,2]
    focus_pad_end_idx = sentinel_indices[:,3] # this is the same as idx but a list of batch size
    focus_mask = torch.arange(position_ids.size(1),device=input_ids.device).expand(len(focus_pad_start_idx), position_ids.size(1)) >= focus_pad_start_idx.unsqueeze(1)
    focus_start_mask = focus_mask <= focus_pad_start_idx.unsqueeze(1)
    focus_end_mask = focus_mask >= focus_pad_start_idx.unsqueeze(1)
    focus_final_mask = focus_start_mask & focus_end_mask
    print(f'focus_start_mask:{focus_start_mask}')
    print(f'fcous_end_mask:{focus_end_mask}') 
    print(f'focus_final_mask:{focus_final_mask}')

    position_ids[cond_final_mask] = pad_id
    position_ids[focus_final_mask] = pad_id
    print(f'paired_position_ids:{position_ids}')

    # for b in range(batch_size):
    # idx = sentinel_indices[b].item()    
    #     if idx > 0:
    #         position_ids[b, :idx] = torch.arange(start = pad_id + 1, end=pad_id+1+idx, dtype=input_ids.dtype, device=input_ids.device)
    #         position_ids[b, idx:] = torch.arange(start=pad_id + 1,end=pad_id + 1+n - idx, dtype=input_ids.dtype, device=input_ids.device)
    #     else:
    #         #a single sequence case
    #         position_ids[b, idx:] = torch.arange(start=pad_id + 1,end=pad_id +1+n, dtype=input_ids.dtype, device=input_ids.device)


    # for b in range(batch_size):
    #     v = input_ids[b]
    #     s = v.size()[0]

    #     idx = sentinel_indices[b].item()  # Convert tensor to integer
    #     print(f'idx:{idx}')
    #     if idx > -1:
    #         # Fill values from 0 to idx in the batch tensor
    #         batch_tensor[b, :idx] = torch.arange(idx, dtype=v.dtype, device=v.device)
    #         batch_tensor_cond[b, :idx] = torch.arange(idx, dtype=v.dtype, device=v.device)
    #         # Fill values from idx to end of the batch tensor with values ranging from 0 to (n-idx)
    #         batch_tensor[b, idx:] = torch.arange(n - idx, dtype=v.dtype, device=v.device)
    #         batch_tensor_focus[b, idx:] = torch.arange(n - idx, dtype=v.dtype, device=v.device)
    #     else:
    #         batch_tensor[b] = torch.arange(s, dtype=v.dtype, device=v.device)
    
    # return batch_tensor,batch_tensor_cond,batch_tensor_focus
    return position_ids
def get_paired_position_embedding(embedding = None,embed_dim=None, position_ids = None, sentinel_indices = None):
    """
    a helper function to create learanble position embeddings for paired sequences
    e.g Assume condition sequence has length 3 and focus sequence has length 2
    <cond_seq></cond_seq><focus_seq></focus_seq> -> [0,1,2,3,0,1,2] instead of the normal [0,1,2,3,4,5,6]
    args:
    embedding (nn.Embedding): the wpe embedding layer created in model_pytorch
    embed_dim (int): The dimension of embedding layer
    position_ids (torch.tensor): the position_ids vector passed in model_pytorch
    sentinel_indices (torch.tensor): the sentinel_indices vector passed in model_pytorch
    sequence_type (str): the type of sequence i.e condition or focus

    returns:
    torch.tensor
    """
    print('position_ids.shape:',position_ids.shape)
    print('embedding:',embedding)
    print(f'setinel_indices:{sentinel_indices}')
    batch_size = position_ids.size(0)
    n = position_ids.size(1)
    idx = sentinel_indices[0][1].item() # get index of end of condition sequence plus condition padding
    # Initialize the batch tensor with zeros
    position_embeddings_shape = list(position_ids.shape)
    position_embeddings_shape.append(embed_dim)
    position_embeddings = torch.zeros(position_embeddings_shape,device=position_ids.device)
    
    if idx > 0: #we have a paired example
        position_embeddings[:, :idx,:] = embedding(position_ids[:, :idx])#condition position embedding
        print(f'paired_position_embeds_cond:{position_embeds.shape} {position_embeds}')
        position_embeddings[:, idx:,:] = embedding(position_ids[:, idx:])#condition position embedding
        print(f'paired_position_embeds:{position_embeds.shape} {position_embeds}')

    else:
        position_embeddings = embedding(position_ids)#condition position embedding
    print(f'psotion:embeddings.shape:{position_embeddings.shape}')
    
    # for b in range(batch_size):
    #     v = position_ids[b]
    #     idx = sentinel_indices[b].item()# Convert tensor to integer
    #     print(f'idx:{idx}')   
    #     if idx > -1:
    #         if sequence_type == 'condition':
    #             batch_tensor[b, :idx] = embedding(v)
    #         else:
    #             # Fill values from idx to end of the batch tensor with values ranging from 0 to (n-idx)
    #             batch_tensor[b, idx:] = embedding(vz)
    #     else:
    #         batch[b] = embedding(vz)
     
      

       
    
    return position_embeddings


# Example usage:
# input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# sentinel_indices = torch.tensor([[2], [3,]])

# result_batch_tensor = create_paired_position_ids(input_ids, sentinel_indices)
# print(result_batch_tensor)
