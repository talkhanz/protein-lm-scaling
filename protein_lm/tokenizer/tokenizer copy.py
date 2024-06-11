import torch
from typing import List, Union, Optional

from rust_trie import Trie 


class Tokenizer:
    def __init__(self, tokens: List[str],tokens_dictionary:Optional[dict] = None, unk_token_id: Optional[int] = None):
        self.ids_to_tokens = tokens
        self.tokens_dictionary = tokens_dictionary
        self.trie = Trie(unk_token_id)
        for token in tokens:
            self.trie.add(token)
        # If unk_token_id is not provided, add <unk> to the end of the tokens list
        if unk_token_id is None:
            self.ids_to_tokens += ["<unk>"]
        self.pad_token_id = self.ids_to_tokens.index("<pad>")
        self.mask_token_id = self.ids_to_tokens.index("<mask>")
        self.vocab_size = len(self.ids_to_tokens)


    def __call__(self,sequences: Union[str, List] ,cond_sequences: Optional[Union[str, List]],focus_sequences: Optional[Union[str, List]],cond_clusters: Optional[Union[str, List]],focus_clusters: Optional[Union[str, List]],decoding_orders: Optional[Union[str, List]], multisequence:Optional[bool]= False, *args, **kwargs):
        if multisequence:
        
            if isinstance(cond_sequences, str) and isinstance(focus_sequences, str) and isinstance(decoding_orders, str) :
                return self.encode_multisequence(sequences, *args, **kwargs)
            else:
                c,f= len(cond_sequences),len(focus_sequences)
                if c == f and f != 0: #all lengths should be non zero and equal to one another
                    return self.batch_encode_multisequence(cond_sequences,focus_sequences,decoding_orders, *args, **kwargs)
                else:
                    raise Exception(f"ERROR: Length mismatch between condition (length={c}) focus (length={f}) sequences and decoding_orders (length={d})")
        else:
            if isinstance(sequences, str):
                return self.encode(sequences, *args, **kwargs)
            else:
                return self.batch_encode(sequences, *args, **kwargs)
    
    def encode(
        self, 
        sequence: str, 
        add_special_tokens: bool = False,
        return_tensor: bool = False,
        max_sequence_length: Optional[int] = None,
    ) -> List[int]:
        if max_sequence_length is not None:
            if add_special_tokens:
                max_sequence_length -= 2
            sequence = sequence[:max_sequence_length]
        if add_special_tokens:
            sequence = "<cls>" + sequence + "<eos>"
        output = self.trie.tokenize(sequence)
        if return_tensor:
            output = torch.tensor(output, dtype=torch.long)
        return output

    def batch_encode(
        self,
        sequences: List[str],
        add_special_tokens: bool = False,
        return_tensors: bool = False,
        max_sequence_length: Optional[int] = None,
    ) -> List[List[int]]:
        output = []
        if max_sequence_length is None and return_tensors:
            max_sequence_length = max([len(sequence) for sequence in sequences])
            if add_special_tokens:
                max_sequence_length += 2
        if max_sequence_length is not None:
            sequences = [
                sequence[:(max_sequence_length - 2) if add_special_tokens else max_sequence_length] 
                for sequence in sequences
            ]
        for sequence in sequences:
            output.append(self.encode(sequence, add_special_tokens, return_tensors))
        if return_tensors:
            tensor_out = torch.full((len(output), max_sequence_length), self.pad_token_id)
            for i, sequence in enumerate(output):
                tensor_out[i, :len(sequence)] = sequence
            output = tensor_out
        return output
    
    def encode_multisequence(
        self, 
        cond_sequence: str,
        focus_sequence: str, 
        cond_sequence_cluster:str,
        focus_sequence_cluster:str,
        decoding_order:str,
        add_special_tokens: bool = False,
        return_tensor: bool = False,
        max_cond_sequence_length: Optional[int] = None,
        max_focus_sequence_length: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ) -> List[int]:
        
        if max_cond_sequence_length is not None:
            if add_special_tokens:
                max_cond_sequence_length -= 2
            cond_sequence = cond_sequence[:max_cond_sequence_length]
        
        if max_focus_sequence_length is not None:
            if add_special_tokens:
                max_focus_sequence_length -= 2
            focus_sequence = focus_sequence[:max_focus_sequence_length]

        c,f = len(cond_sequence),len(focus_sequence)
        if max_cond_sequence_length < c or max_focus_sequence_length < f:
            print(f'WARNING: one of condition or focus sequence has been specified a maximum length smaller than actual sequence length')
        cond_pad_length = max_cond_sequence_length - c - 2 #subtracting the two start and end tokens
        cond_pad_sequence = "".join(["<pad>" for i in range(cond_pad_length)])
        decoding_order_to_focus_sequence = {
            'lr': focus_sequence, 'rl': focus_sequence[::-1],
            'fim': self.encode_fim(focus_sequence) , 'mo': self.encode_mo(focus_sequence),
        }
        if add_special_tokens:
            if c == 0: #only a single sequence, we dont have a condition sequence 
                sequence = "<cls>" + focus_sequence + "<eos>"
            else: #paired sequence, we have both a condition and focus sequence
                cluster_checks = ['50','90','100']
                decoding_order_checks = ['lr','rl','fim','mo']
                cond_modified_sequence = ''
                focus_modified_sequence = ''
                for check in cluster_checks:
                    if check in cond_sequence_cluster:
                        for check2 in decoding_order_checks:
                            cond_start_token = [t for t in self.tokens_dictionary['cond_start_tokens'] if check in cond_sequence_cluster][0]
                            cond_end_token = self.tokens_dictionary['cond_end_tokens'][0]
                            cond_modified_sequence = cond_start_token + cond_sequence + cond_pad_sequence + cond_end_token
                    if check in focus_sequence_cluster:
                        for check2 in decoding_order_checks:
                            focus_end_token = self.tokens_dictionary['focus_end_tokens'][0]
                            
                            if decoding_order in ['mo','fim']:
                                focus_pad_length = max_focus_sequence_length - f - 2  - 2  #subtracting the two start and end tokens and middle and suffix tokens
                                focus_pad_sequence =  "".join(["<pad>" for i in range(focus_pad_length)])
                                prefix_start_token = [t for t in self.tokens_dictionary['focus_start_tokens'] if check in focus_sequence_cluster and check2 in decoding_order+'-l'][0]
                                middle_start_token = [t for t in self.tokens_dictionary['focus_start_tokens'] if check in focus_sequence_cluster and check2 in decoding_order+'-m'][0]
                                suffix_start_token = [t for t in self.tokens_dictionary['focus_start_tokens'] if check in focus_sequence_cluster and check2 in decoding_order+'-r'][0]
                                prefix_sequence,middle_sequence,suffix_sequence = decoding_order_to_focus_sequence[decoding_order]
                                focus_sequence_modified =  prefix_start_token + prefix_sequence + middle_start_token + middle_sequence + suffix_start_token + suffix_sequence + focus_end_token + focus_pad_sequence
                            else:
                                focus_pad_length = max_focus_sequence_length - f - 2   #subtracting the two start and end tokens
                                focus_pad_sequence =  "".join(["<pad>" for i in range(focus_pad_length)])
                                focus_start_token = [t for t in self.tokens_dictionary['focus_start_tokens'] if check in focus_sequence_cluster and check2 in decoding_order][0]
                                focus_sequence = decoding_order_to_focus_sequence[decoding_order]
                                focus_modified_sequence = focus_start_token + focus_sequence + focus_end_token + focus_pad_sequence 
                
                sequence = cond_modified_sequence + focus_modified_sequence
                


        print(f'INFO: tokenizing sequence:{sequence}')
        output = self.trie.tokenize(sequence)
        if return_tensor:
            output = torch.tensor(output, dtype=torch.long)
        return output

    def batch_encode_multisequence(
        self,
        cond_sequences: List[str],
        focus_sequences: List[str],
        cond_sequence_clusters: List[str],
        focus_sequences_clusters: List[str],
        decoding_orders: List[str],
        add_special_tokens: bool = False,
        return_tensors: bool = False,
        max_cond_sequence_length: Optional[int] = None,
        max_focus_sequence_length: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ) -> List[List[int]]:

        output = []
        if max_cond_sequence_length is None and return_tensors:
            max_cond_sequence_length = max([len(sequence) for sequence in cond_sequences])
            if add_special_tokens:
                max_cond_sequence_length += 2
        if max_cond_sequence_length is not None:
            cond_sequences = [
                sequence[:(max_cond_sequence_length - 2) if add_special_tokens else max_cond_sequence_length] 
                for sequence in cond_sequences
            ]
        
        if max_focus_sequence_length is None and return_tensors:
            max_focus_sequence_length = max([len(sequence) for sequence in focus_sequences])
            if add_special_tokens:
                max_focus_sequence_length += 2
        if max_focus_sequence_length is not None:
            focus_sequences = [
                sequence[:(max_focus_sequence_length - 2) if add_special_tokens else max_focus_sequence_length] 
                for sequence in focus_sequences
            ]
        
        
        for i in range(len(cond_sequences)):
            output.append(self.encode_multisequence(cond_sequences[i],focus_sequences[i],cond_sequence_clusters[i],focus_sequence_clusters[i],decoding_orders[i], add_special_tokens, return_tensors,max_cond_sequence_length,max_focus_sequence_length))
        if return_tensors:
            tensor_out = torch.full((len(output), max_sequence_length), self.pad_token_id)
            for i, sequence in enumerate(output):
                tensor_out[i, :len(sequence)] = sequence
            output = tensor_out
        return output

    def encode_fim(self, sequence:str) -> str:
        #expect a pure sequence without padding/sentinel tokens
        s = len(sequence)
        indices = list(range(s))
        feasible_middle_indices = list(range(int(0.2 * s),int(0.8 * s))) #ensure the start of middle index will be atleast between the middle 60% (after the first 20% tokens and before last 20% tokens ) e.g <20%>|<60%>|<20%>
        feasible_middle_lengths = list(range(max(int(0.02 * s), 1), int(0.5 * s))) #length can be only betwee 2% of sequence_length or 1 and 50% of sequence 
        
        middle_index =  np.random.choice(feasible_middle_indices) # randomly pick a middle index
        middle_length =  np.random.choice(feasible_middle_lengths) # randomly pick length of middle region
        prefix_sequence = sequence[:middle_index]
        middle_sequence = sequence[middle_index:middle_index + middle_length]
        suffix_sequence = sequence[middle_index + middle_length:]
        return prefix_sequence,middle_sequence,suffix_sequence

    def encode_mo(self, sequence:str) -> str:
        #expect a pure sequence without padding/sentinel tokens
        s = len(sequence)
        indices = list(range(s))
        feasible_middle_indices = list(range(int(0.2 * s),int(0.8 * s))) #ensure the start of middle index will be atleast between the middle 60% (after the first 20% tokens and before last 20% tokens ) e.g <20%>|<60%>|<20%>
        #considering a slightly longer middle length here since we will be predicting both prefix and suffix regions
        feasible_middle_lengths = list(range(max(int(0.2 * s), 1), int(0.5 * s))) #length can be only betwee 20% of sequence_length or 1 and 50% of sequence 
        
        middle_index =  np.random.choice(feasible_middle_indices) # randomly pick a middle index
        middle_length =  np.random.choice(feasible_middle_lengths) # randomly pick length of middle region
        prefix_sequence = sequence[:middle_index]
        middle_sequence = sequence[middle_index:middle_index + middle_length]
        suffix_sequence = sequence[middle_index + middle_length:]
        return prefix_sequence,middle_sequence,suffix_sequence

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.ids_to_tokens[idx] for idx in tokens])
    


class EsmTokenizer(Tokenizer):
    def __init__(self):
        tokens = [
            "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", 
            "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", 
            "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", 
            "Z", "O", ".", "-", "<null_1>", "<mask>"
        ]
        super().__init__(tokens, unk_token_id=3)



class AptTokenizer(Tokenizer):
    def __init__(self):
        # For our own tokenizers, we don't need to explicitly add the <unk> token
        # because it gets added as the last token in the tokens list
        # I've also removed X so that it gets translated to <unk>
        self.tokens_dictionary = {
            "uncond_start_tokens": ["<cls>"],
            "cond_start_tokens": ["<bc-30>","<bc-50>","<bc-90>","<bc-100>","<bc-str>"],
            "tokens": ["L", "A", "G", "V", 
            "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", 
            "F", "Y", "M", "H", "W", "C", "B", "U", "Z", "O"],
            "cond_end_tokens":  ["<eoc>"],
            "special_tokens": ["<pad>","<mask>"],
            "focus_start_tokens" : ["<bf-30-lr>","<bf-30-rl>","<bf-30-fim-l>","<bf-30-fim-r>","<bf-30-fim-m>","<bf-30-mo-l>","<bf-30-mo-r>","<bf-30-mo-m>",
                                    "<bf-50-lr>","<bf-50-rl>","<bf-50-fim-l>","<bf-50-fim-r>","<bf-50-fim-m>","<bf-50-mo-l>","<bf-50-mo-r>","<bf-50-mo-m>",
                                    "<bf-90-lr>","<bf-90-rl>","<bf-90-fim-l>","<bf-90-fim-r>","<bf-90-fim-m>","<bf-90-mo-l>","<bf-90-mo-r>","<bf-90-mo-m>",
                                    "<bf-100-lr>","<bf-100-rl>","<bf-100-fim-l>","<bf-100-fim-r>","<bf-100-fim-m>","<bf-100-mo-l>","<bf-100-mo-r>","<bf-100-mo-m>",
                                    ],
            "focus_end_tokens": ["<eof>"],
            "end_tokens": ["<eos>"],
        }
        self.tokens = [token for key,tokens in self.tokens_dictionary.items() for token in tokens]
        super().__init__(self.tokens,tokens_dictionary=tokens_dictionary)