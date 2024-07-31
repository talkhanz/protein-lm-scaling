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


def reorder_matrix(A, sentinel_indices,decoding_orders=[]):
    """
    reorders fim cases from prefix,middle,suffix to prefix,suffix,middle
    reorders mo cases from prefix,middle,suffix to middle,prefix,suffix

    args:
    A (2D matrix): the position/input ids vector
    sentinel_indices (2D matrix): sentinel indices specifying the location of cond/focus padding start/end and prefix/middle/suffix regions
    decoding_orders (list str)
    returns:
        torch.tensor
    """
    def reorder_fim(A,sentinel_indices,N,i):
        result = torch.zeros_like(A[i])
        prefix_idx,middle_idx,suffix_idx,focus_pad_start_idx = sentinel_indices[i,4],sentinel_indices[i,5],sentinel_indices[i,6],sentinel_indices[i,2]
        check1 = all([prefix_idx == 0, middle_idx == 0, suffix_idx == 0])
        check2 = any([middle_idx == 0, suffix_idx == 0])
        check3 = any([middle_idx <= prefix_idx, suffix_idx <= middle_idx,suffix_idx <= prefix_idx])
        if check1 or check2 or check3:
            #prefix/middle/suffix don't exist
            return A[i]
        middle_length,suffix_length = suffix_idx - middle_idx,  focus_pad_start_idx - suffix_idx
        # print(f'middle_length,suffix_length:{middle_length},{suffix_length}')
        middle_new_indices =  torch.arange(start=middle_idx+suffix_length,end=focus_pad_start_idx)
        suffix_new_indices =  torch.arange(start=middle_idx,end=middle_idx+suffix_length)
        # print(f'middle_new_indices:{middle_new_indices}')
        # print(f'suffix_new_indices:{suffix_new_indices}')
        
        shuffle_indices = torch.arange(N)
        # print(f'shuffle_indices_before:{shuffle_indices}')
        shuffle_indices[middle_idx:middle_idx+middle_length] = middle_new_indices
        shuffle_indices[middle_idx+middle_length: middle_idx+middle_length + suffix_length] = suffix_new_indices
        # print(f'shuffle_indices_after:{shuffle_indices}')
        #shuffle_indices -> [3,2,1]  A -> [1,2,3] -> A_shuffled ->[3,2,1,]
        #FIM: A -> [1,2,3,,4,5,6,7,8,9] -> shuffle_indices -> [1,2,3,7,8,9,4,5,6] A_shuffled:[1,2,3,7,8,9,4,5,6] preifx,suffix,middle
        result.scatter_(0, shuffle_indices, A[i])
        return result
    def reorder_mo(A,sentinel_indices,N,i):
        result = torch.zeros_like(A[i])
        prefix_idx,middle_idx,suffix_idx,focus_pad_start_idx = sentinel_indices[i,4],sentinel_indices[i,5],sentinel_indices[i,6],sentinel_indices[i,2]
        check1 = all([prefix_idx == 0, middle_idx == 0, suffix_idx == 0])
        check2 = any([middle_idx == 0, suffix_idx == 0])
        check3 = any([middle_idx <= prefix_idx, suffix_idx <= middle_idx,suffix_idx <= prefix_idx])
        if check1 or check2 or check3:
            #prefix/middle/suffix don't exist
            return A[i]
        prefix_length,middle_length,suffix_length = middle_idx - prefix_idx,suffix_idx - middle_idx,  focus_pad_start_idx - suffix_idx
        # print(f'prefix_length,middle_length,suffix_length:{prefix_length},{middle_length},{suffix_length}')

        middle_new_indices =  torch.arange(start=prefix_idx,end=prefix_idx + middle_length)
        prefix_new_indices =  torch.arange(start=prefix_idx + middle_length ,end=prefix_idx + middle_length +prefix_length)
        # print(f'middle_new_indices:{middle_new_indices}')
        # print(f'preffix_new_indices:{prefix_new_indices}')
        
        shuffle_indices = torch.arange(N)
        # print(f'shuffle_indices_before:{shuffle_indices}')
        shuffle_indices[middle_idx:middle_idx+middle_length] = middle_new_indices
        shuffle_indices[prefix_idx: prefix_idx + prefix_length] = prefix_new_indices
        # print(f'shuffle_indices_after:{shuffle_indices}')
        #MO: A -> [1,2,3,,4,5,6,7,8,9] -> shuffle_indices -> [4,5,6,1,2,3,7,8,9] A_shuffled:[4,5,6.1,2,3,7,8,9] middle,preifx,suffix,
        result.scatter_(0, shuffle_indices, A[i])
        return result
    batch_size = A.shape[0]
    N = A.shape[-1]
    result = torch.zeros_like(A)
    for i in range(batch_size):
        if decoding_orders[i] == 'fim':
            result[i] = reorder_fim(A,sentinel_indices,N,i)
        elif decoding_orders[i] == 'mo':
            result[i] = reorder_mo(A,sentinel_indices,N,i)
    return result

def split_matrix(matrix=None, sentinel_indices = None):
    rows = torch.arange(matrix.shape[0]).unsqueeze(1)
    cols = torch.arange(matrix.shape[1])
    before_mask = cols < sentinel_indices.unsqueeze(1)

    # Select elements before the index
    before_split = matrix[before_mask].split(sentinel_indices.tolist())

    # Create a mask for selecting elements after the index
    after_mask = cols >= sentinel_indices.unsqueeze(1)

    # Select elements after the index
    after_split = matrix[after_mask].split((matrix.shape[1] - sentinel_indices).tolist())

    # Convert the lists of tensors back to tensors if needed
    before_split = torch.nn.utils.rnn.pad_sequence(before_split, batch_first=True, padding_value=0)
    after_split = torch.nn.utils.rnn.pad_sequence(after_split, batch_first=True, padding_value=0)
    return before_split,after_split

def get_paired_hidden_states(hidden_states=None,sentinel_indices=None):
    """
    separates condition and focus hidden_states  from the complete hidden_states vector

    args:
    hidden_states (torch.tensor): the hidden representations of input embeds
    sentinel_indices (torch.tensor): the sentinel_indices vector passed in model_pytorch
    """
    print('def get_paired_hidden_states(hidden_states=None,sentinel_indices=None):')
    print('hidden_states.shape:',hidden_states.shape)
    print('sentinel_indices:',sentinel_indices.shape)

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
def reoorder_ids(input_ids=None,positions_ids = None, sentinel_indices = None, decoding_order = 'l2r'):
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
    
    return position_ids


def get_paired_position_embedding(embedding = None,embedding_condition = None,embed_dim=None, position_ids = None, sentinel_indices = None):
    """
    a helper function to create learanble position embeddings for paired sequences
    e.g Assume condition sequence has length 3 and focus sequence has length 2
    <cond_seq></cond_seq><focus_seq></focus_seq> -> [0,1,2,3,0,1,2] instead of the normal [0,1,2,3,4,5,6]
    args:
    embedding (nn.Embedding): the wpe embedding layer created in model_pytorch
    embedding_cond (nn.Embedding): the wpe_conditioning embedding layer created in model_pytorch
    embed_dim (int): The dimension of embedding layer
    position_ids (torch.tensor): the position_ids vector passed in model_pytorch
    sentinel_indices (torch.tensor): the sentinel_indices vector passed in model_pytorch
    sequence_type (str): the type of sequence i.e condition or focus

    returns:
    torch.tensor
    """
    print("#########get_paired_position_embedding#########")
    # print('position_ids.shape:',position_ids.shape)
    # print('embedding:',embedding)
    print(f'setinel_indices:{sentinel_indices}')
    batch_size = position_ids.size(0)
    n = position_ids.size(1)
    idx = sentinel_indices[0][1].item() # get index of end of condition sequence plus condition padding
    # Initialize the batch tensor with zeros
    position_embeddings_shape = list(position_ids.shape)
    position_embeddings_shape.append(embed_dim)
    position_embeddings = torch.zeros(position_embeddings_shape,device=position_ids.device)

    # print(f'cond_mask:',cond_mask.shape,cond_mask)
    # print(f'focus_mask:',focus_mask.shape,focus_mask)
    position_embeddings[:, :idx] = embedding_condition(position_ids[:, :idx])#condition position embedding
    # print(f'psotion_embeddings_cond.shape:{position_embeddings.shape}')
    position_embeddings[:, idx:] = embedding(position_ids[:, idx:])#focus position embedding
    
    print(f'position_embeddings.shape:{position_embeddings.shape}')
    print(f'position_embeddings:{position_embeddings}')
    return position_embeddings



