import torch
import numpy as np
import math
def create_attention_mask(tokenized_sequence = [], padding_idx = [],focus_idx=[],span_length = 5,sequence_mask_fraction= 0.25,mask_type = "",mask_id = None,return_tensor = False):
    def isin(number,pairs):
        
        print(f'pairs:{pairs}')
        if len(pairs) == 0 :
            return False
        if isinstance(pairs,list):
            pairs = np.array(pairs)
        within_start = pairs[:, 0] <= number
        within_end = pairs[:, 1] >= number

        # Combine the conditions to find pairs that contain the number
        within_pair = np.logical_and(within_start, within_end)

        # Check if any pair contains the number
        result = np.any(within_pair)
        return result
    print(f'create_attention_mask(tokenized_sequence = {tokenized_sequence}, padding_idx = {padding_idx},focus_idx={focus_idx},span_length = {span_length},sequence_mask_fraction= {sequence_mask_fraction},mask_type = {mask_type},mask_id = {mask_id},return_tensor = {return_tensor}):')    
    sequence_length = len(tokenized_sequence)
    mask = torch.ones(sequence_length)
    mask[padding_idx] = 0
    indices = [i for i in range(1,sequence_length) if i not in padding_idx + focus_idx]
    print(f'indices:{indices}')
    indices.sort()
    mask_budget = int(np.ceil((len(indices) * sequence_mask_fraction)))
    print(f'mask_budget:{mask_budget}')
    available_mask_budget = mask_budget
    masked_output = list(tokenized_sequence)
    if return_tensor:
        masked_output = tokenized_sequence.clone()

    if mask_type == "random": # selects random indices
        mask_idx = np.random.choice(indices,size=mask_budget)
        print(f'mask_idx:{mask_idx}')
        mask[mask_idx] = 0
        masked_output[mask_idx] = mask_id
    elif mask_type == "random_span": # creates a single span of length condition_sequence AA tokens * fraction
        if span_length > mask_budget:
            prev_span_length = span_length
            span_length = max(1,int(mask_budget - 1))
            print(f'WARNING: Provided span_length:{prev_span_length} is greater than available masking budget of {mask_budget}. Setting span length to:{span_length}')

        span_start_index = np.random.choice(indices)
        while span_start_index + mask_budget > indices[-1]:
            span_start_index = np.random.choice(indices)
        span_end_index = span_start_index + mask_budget
        mask[span_start_index:span_end_index] = 0
        masked_output[span_start_index:span_end_index] = mask_id
            
        

    elif mask_type == "random_multi_span": #creates multiple spans
        if span_length > mask_budget:
            prev_span_length = span_length
            span_length = max(1,int(mask_budget) / 4)
            print(f'WARNING: Provided span_length:{prev_span_length} is greater than available masking budget of {mask_budget}. Setting span length to:{span_length}')

        used_span_indices = []
        while available_mask_budget > 0:
            span_start_index = np.random.choice(indices)
            span_min,span_max = max(1,span_length -2), min(span_length + 2, mask_budget)
            sampled_span_length = int(np.random.uniform(span_min, span_max))
            while span_start_index + sampled_span_length > indices[-1]:
                if not isin(span_start_index,used_span_indices): #make sure the current span_start_index does not lie in past span interval
                    span_start_index = np.random.choice(indices)
            
            span_end_index = span_start_index + sampled_span_length
            used_span_indices.append([span_start_index,span_end_index])
            mask[span_start_index:span_end_index] = 0
            masked_output[span_start_index:span_end_index] = mask_id
            span_length_actual = span_end_index - span_start_index
            available_mask_budget = available_mask_budget - span_length_actual
 
    elif mask_type == "both":
        pass

    print('attention_')
    return masked_output,mask

# def getmask_subset_with_fraction(self, mask: torch.Tensor,
#                                       prob: float) -> torch.Tensor:
#     '''
#     Probability for mask=True, rounds up number of residues
#     '''
#     batch, seq_len, device = mask.shape, mask.device
#     num_tokens = mask.sum(dim=-1, keepdim=True) # number of non-masked tokens in each sequence
#     num_to_mask = (num_tokens *prob).floor().type(torch.int64).squeeze(1).tolist() # number of tokens to mask in each sequence
#     max_masked = math.ceil(prob * num_tokens.max() + 1) 
#     sampled_indices = -torch.ones((batch, max_masked), 
#                                     dtype=torch.int64, device=device)

#     # select random indices to mask
#     for i in range(batch):
#         rand = torch.rand((seq_len), device=device).masked_fill(~mask[i], -1e9)
#         , sampled_indices[i,:num_to_mask[i]] = rand.topk(num_to_mask[i], dim=-1)

#     sampled_indices = sampled_indices + 1  # padding is 0 allow excess scatter to index 0
#     new_mask = torch.zeros((batch, seq_len + 1), device=device)
#     new_mask.scatter_(-1, sampled_indices, 1)
#     return new_mask[:, 1:].bool()  # index 0 removed

# def train_masking(self, tokens: torch.Tensor,
#                     mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     '''Masking as described by ESM
#     Because sum of probabilities of noise types add up to 1, treat as fraction instead
#     '''
#     noise_mask = self.get_mask_subset_with_fraction(mask, self.mask_prob)
#     if self.simple_masking_only:
#         mask_mask = noise_mask
#         noised_tokens = torch.where(mask_mask, self.tokenizer.mask_idx, tokens)
#     return noised_tokens,noise_mask