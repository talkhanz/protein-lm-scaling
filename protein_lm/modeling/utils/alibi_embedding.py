import math
import torch
from typing import Optional

def get_slopes(n):
    """
    Function to compute the m constant for each attention head. Code has been adapted from the official ALiBi codebase at:
    https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py
    """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   
    else:                                                 
        closest_power_of_2 = 2**math.floor(math.log2(n))   
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

 
def create_alibi_tensor(attn_heads,maxpos):
    slopes = torch.Tensor(get_slopes(attn_heads))
    #The softmax operation is invariant to translation, and bias functions used are always linear. 
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
    alibi = alibi.view(attn_heads, 1, maxpos)
    print(f'alibi:{alibi.shape}')
    alibi = alibi.repeat(1, 1, 1)  # batch_size, 1, 1
    print(f'alibi:{alibi.shape}')
    return alibi
    




def build_alibi_bias(
    n_heads: int = 3,
    seq_len: int = 8,
    full: bool = False,
    alibi_bias_max: int = 8,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    def gen_slopes(
        n_heads: int,
        alibi_bias_max: int = 8,
        device: Optional[torch.device] = None,
        return_1d: bool = False,
    ) -> torch.Tensor:
        _n_heads = 2**math.ceil(math.log2(n_heads))
        m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
        m = m.mul(alibi_bias_max / _n_heads)
        slopes = (1. / torch.pow(2, m))

        if _n_heads != n_heads:
            # if n_heads is not a power of two,
            # Huggingface and FasterTransformer calculate slopes normally,
            # then return this strided concatenation of slopes
            slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
        if return_1d:
            return slopes
        return slopes.view(1, n_heads, 1, 1)
    print(f'build_alibi_bias->seq_len:{seq_len}')
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32,
                                device=device).view(1, 1, 1, seq_len)
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is broadcast to the appropriate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len,
            1,
            dtype=torch.int32,
            device=device,
        ).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)

    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)

