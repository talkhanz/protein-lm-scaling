from typing import Dict, Literal,Optional
from pydantic import BaseModel
from protein_lm.tokenizer.tokenizer import AptTokenizer


class TokenizerConfig(BaseModel):
    tokenizer_type: Literal["APT"]
    mask_type: Optional[Literal['random','random_span','random_multi_span']] 
    sequence_mask_fraction: Optional[dict]
    

def get_tokenizer(config_dict: Dict):
    config = TokenizerConfig(**config_dict)
    if config.tokenizer_type == "APT":
        tokenizer_constructor = AptTokenizer
    else:
        raise ValueError(f"Invalid tokenizer_type {config.tokenizer_type}")

    return tokenizer_constructor(sequence_mask_fraction=config.sequence_mask_fraction,mask_type = config.mask_type)
