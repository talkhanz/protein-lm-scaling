from transformers import GPT2Config
from typing import Literal
from protein_lm.tokenizer.tokenizer import AptTokenizer


class APTConfig(GPT2Config):
    """
    Config subclass for Autoregressive Protein Transformer (APT) model architecture.
    """

    def __init__(
        self,
        position_embedding: Literal["alibi", "learned", "rope", "rerope", "linear_rope_scaling", "dynamic_rope_scaling"]="learned",
        tokenizer=None,
        max_sequence_length = 1024,
        attn_type="standard",
        position_embedding_type = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nn_model_type = "APT"
        self.position_embedding = position_embedding
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.attn_type = attn_type
        self.position_embedding_type = position_embedding_type
        self.vocab_size = getattr(AptTokenizer(),'vocab_size')
        self.pad_id = getattr(AptTokenizer(),'tokens').index("<pad>")

