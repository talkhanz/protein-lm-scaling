import torch
import numpy as np

def create_attention_mask(sequence_length=2048,padding_idx = [],focus_idx=[],cond_sequence_mask_fraction=0.25):
    mask = torch.zeros(sequence_length)
    mask[padding_idx] = 1
    indices = [i for i in range(sequence_length) if i not in padding_idx + focus_idx]
    num_indices = int(len(indices) * cond_sequence_mask_fraction)
    n = len(num_indices)
    mask_idx = np.random.choice(indices,size=n)
    mask[mask_idx] = 1
    return mask

def getmask_subset_with_fraction(self, mask: torch.Tensor,
                                      prob: float) -> torch.Tensor:
    '''
    Probability for mask=True, rounds up number of residues
    '''
    batch, seq_len, device = mask.shape, mask.device
    num_tokens = mask.sum(dim=-1, keepdim=True) # number of non-masked tokens in each sequence
    num_to_mask = (num_tokens *prob).floor().type(torch.int64).squeeze(1).tolist() # number of tokens to mask in each sequence
    max_masked = math.ceil(prob * num_tokens.max() + 1) 
    sampled_indices = -torch.ones((batch, max_masked), 
                                    dtype=torch.int64, device=device)

    # select random indices to mask
    for i in range(batch):
        rand = torch.rand((seq_len), device=device).masked_fill(~mask[i], -1e9)
        , sampled_indices[i,:num_to_mask[i]] = rand.topk(num_to_mask[i], dim=-1)

    sampled_indices = sampled_indices + 1  # padding is 0 allow excess scatter to index 0
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()  # index 0 removed

def train_masking(self, tokens: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Masking as described by ESM
    Because sum of probabilities of noise types add up to 1, treat as fraction instead
    '''
    noise_mask = self.get_mask_subset_with_fraction(mask, self.mask_prob)
    if self.simple_masking_only:
        mask_mask = noise_mask
        noised_tokens = torch.where(mask_mask, self.tokenizer.mask_idx, tokens)
    return noised_tokens