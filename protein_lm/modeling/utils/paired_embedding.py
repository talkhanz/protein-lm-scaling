import torch

def create_paired_block_diagonal(A=None, B = None):
    """
    takes two matrices of same batch size K,different lengths (N/M each corresponding to sequence) and 
    the model dimension X
    A (torch.tensor)
    B (torch.tensor)
    returns:
    torch.tensor
    """
    K,N,X = A.shape
    K,M,X = B.shape
    block_diag = torch.cat([
        torch.cat([A, torch.zeros(K,N, X)], dim=2),
        torch.cat([torch.zeros(K,M, X), B], dim=2)
    ], dim=1)

    print("Matrix A:",A.shape)

    print(A)
    print("\nMatrix B:",B.shape)
    print(B)
    print("\nBlock Diagonal Matrix:",block_diag.shape)
    print(block_diag)
    return block_diag
def get_paired_hidden_states(hidden_states=None,sentinel_indices=None):
    """
    separates condition and focus hidden_states  from the complete hidden_states vector

    args:
    hidden_states (torch.tensor): the hidden representations of input embeds
    sentinel_indices (torch.tensor): the sentinel_indices vector passed in model_pytorch
    """
    print('def get_paired_hidden_states(hidden_states=None,sentinel_indices=None):')
    print('hidden_states.shape:',hidden_states.shape)
    print('sentinel_indices:',sentinel_indices)

    idx = sentinel_indices[0][1].item() # get index of end of condition sequence plus condition padding
    print(f'idx:{idx}')
    hidden_cond,hidden_focus = hidden_states[:,:idx].detach().clone(),hidden_states[:,idx:].detach().clone()
    print("hidden_cond,hidden_focus:",hidden_cond.shape,hidden_focus.shape)
    return hidden_cond,hidden_focus

def get_paired_position_ids(position_ids=None,sentinel_indices=None):
    """
    separates condition and focus position ids from the complete position_ids vector
    args:
    position_ids (torch.tensor): position_ids
    sentinel_indices (torch.tensor): the sentinel_indices vector passed in model_pytorch
    """
    print('def get_paired_position_ids(position_ids=None,sentinel_indices=None):')
    print('position_ids.shape:',position_ids.shape)
    print('sentinel_indices:',sentinel_indices)
    
    idx = sentinel_indices[0][1].item() # get index of end of condition sequence plus condition padding
    print(f'idx:{idx}')
    position_ids_cond,position_ids_focus = position_ids[:,:idx].detach().clone(),position_ids[:,idx:].detach().clone() 
    print("position_ids_cond,position_ids_focus:",position_ids_cond.shape,position_ids_focus.shape)
    return position_ids_cond,position_ids_focus
def reoorder_position_ids(positions_ids = None, sentinel_indices = None, decoding_order = 'l2r'):
    pass
        
def create_paired_position_ids(input_ids, sentinel_indices,pad_id=0):
    """
    a helper function to create aboslute position encodings for paired sequences
    e.g Assume condition sequence has length 3 and focus sequence has length 2
    <cond_seq></cond_seq><focus_seq></focus_seq> we want [0,1,2,3,0,1,2] instead of the normal [0,1,2,3,4,5,6]
    args: abcd<eoc><bof>defg<eof> -> [0,1,2,3,1,2,3,4]
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
    
    # position_ids[:, :idx] = torch.arange(start = pad_id + 1, end=pad_id+1+idx, dtype=input_ids.dtype, device=input_ids.device)
    # position_ids[:, idx:] = torch.arange(start = pad_id + 1, end=pad_id+1+n-idx, dtype=input_ids.dtype, device=input_ids.device)
    
    position_ids[:, :idx] = torch.arange(start = 0, end=idx, dtype=input_ids.dtype, device=input_ids.device)
    position_ids[:, idx:] = torch.arange(start = 0, end=n-idx, dtype=input_ids.dtype, device=input_ids.device)
    
    cond_pad_start_idx = sentinel_indices[:,0]
    cond_pad_end_idx = sentinel_indices[:,1] # this is the same as idx but a list of batch size
    cond_mask = torch.arange(position_ids.size(1),device=input_ids.device).expand(len(cond_pad_start_idx), position_ids.size(1)) >= cond_pad_start_idx.unsqueeze(1)
    cond_start_mask = cond_mask <= cond_pad_start_idx.unsqueeze(1)
    cond_end_mask = cond_mask >= cond_pad_start_idx.unsqueeze(1)
    cond_mask = cond_start_mask & cond_end_mask 
    # print(f'cond_start_mask:{cond_start_mask}')
    # print(f'cond_end_mask:{cond_end_mask}')
    # print(f'cond_final_mask:{cond_mask}')

    focus_pad_start_idx = sentinel_indices[:,2]
    focus_pad_end_idx = sentinel_indices[:,3] # this is the same as idx but a list of batch size
    focus_mask = torch.arange(position_ids.size(1),device=input_ids.device).expand(len(focus_pad_start_idx), position_ids.size(1)) >= focus_pad_start_idx.unsqueeze(1)
    focus_start_mask = focus_mask <= focus_pad_start_idx.unsqueeze(1)
    focus_end_mask = focus_mask >= focus_pad_start_idx.unsqueeze(1)
    focus_mask = focus_start_mask & focus_end_mask
   
    position_ids[cond_mask] = pad_id
    position_ids[focus_mask] = pad_id
    # print(f'paired_position_ids:{position_ids}')

    return position_ids,cond_mask,focus_mask
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
    # print('position_ids.shape:',position_ids.shape)
    # print('embedding:',embedding)
    print(f'setinel_indices:{sentinel_indices}')
    batch_size = position_ids.size(0)
    n = position_ids.size(1)

    # Initialize the batch tensor with zeros
    position_embeddings_shape = list(position_ids.shape)
    position_embeddings_shape.append(embed_dim)
    position_embeddings = torch.zeros(position_embeddings_shape,device=position_ids.device)

    cond_pad_start_idx = sentinel_indices[:,0]
    cond_pad_end_idx = sentinel_indices[:,1] # this is the same as idx but a list of batch size
    cond_mask = torch.arange(position_ids.size(1),device=position_ids.device).expand(len(cond_pad_start_idx), position_ids.size(1)) >= cond_pad_start_idx.unsqueeze(1)
    cond_start_mask = cond_mask <= cond_pad_start_idx.unsqueeze(1)
    cond_end_mask = cond_mask >= cond_pad_start_idx.unsqueeze(1)
    cond_final_mask = cond_start_mask & cond_end_mask 
    
    focus_pad_start_idx = sentinel_indices[:,2]
    focus_pad_end_idx = sentinel_indices[:,3] # this is the same as idx but a list of batch size
    focus_mask = torch.arange(position_ids.size(1),device=position_ids.device).expand(len(focus_pad_start_idx), position_ids.size(1)) >= focus_pad_start_idx.unsqueeze(1)
    focus_start_mask = focus_mask <= focus_pad_start_idx.unsqueeze(1)
    focus_end_mask = focus_mask >= focus_pad_start_idx.unsqueeze(1)
    focus_final_mask = focus_start_mask & focus_end_mask
    # print(f'cond_mask:',cond_mask.shape,cond_mask)
    # print(f'focus_mask:',focus_mask.shape,focus_mask)
    position_embeddings[cond_mask,:] = embedding(position_ids[cond_mask])#condition position embedding
    # print(f'psotion_embeddings_cond.shape:{position_embeddings.shape}')
    position_embeddings[focus_mask,:] = embedding(position_ids[focus_mask])#condition position embedding
    
    print(f'psotion:embeddings.shape:{position_embeddings.shape}')
    
    
 
     
      

       
    
    return position_embeddings,cond_mask,focus_mask



