import torch
from typing import List, Union, Optional

from rust_trie import Trie 


class Tokenizer:
    def __init__(self, tokens: List[str],sentinel_tokens:List[str],tokens_dictionary:dict, unk_token_id: Optional[int] = None):
        self.ids_to_tokens = tokens
        self.ids_to_sentinel_tokens = sentinel_tokens
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

    def __call__(self, sequences: Union[str, List],is_paired_dataset:bool=False,*args, **kwargs):
        if isinstance(sequences, str):
            if is_paired_dataset:
                return self.encode_pair(sequences, *args, **kwargs)
            else:
                return self.encode(sequences, *args, **kwargs)
        else:
            if is_paired_dataset:
                return self.batch_encode_pair(sequences, *args, **kwargs)
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
    
    def encode_pair(
        self, 
        sequence: str, 
        return_tensor: bool = False,
        max_cond_sequence_length: Optional[int] = 10,
        max_focus_sequence_length: Optional[int] = 10,
        max_sequence_length: Optional[int] = 20,
    ) -> List[int]: #specific encode function for sequences from PairedDataset
        def get_index(sequence,token):    
            try:
                index = sequence.index(token)
            except Exception as e:
                # print('get_index_exception:')
                # print(e)
                index = -1
            # print('get_index(sequence,token)')
            # print(sequence,token,index)
            return index
        
        token_index = []
        old_sequence = sequence
        sequence =  self.trie.tokenize(sequence)
        for token in self.ids_to_sentinel_tokens:
            index = get_index(sequence[:max_sequence_length],token)
            if index > -1:
                token_index.append(index)
        print('sequence:',sequence,len(sequence),token_index)
        
        if len(token_index):
            sentinel_token = sequence[token_index[0]]
            print(f'sentinel_token:{sentinel_token}')
            splits = sequence.split(sentinel_token)
            
            
            condition_split_with_sentinel = splits[0]
            if len(condition_split_with_sentinel) > 0:
                focus_split_with_sentinel = splits[1]
                print(f'condition_split_with_sentinel:{condition_split_with_sentinel}')
                print(f'focus_split_with_sentinel:{focus_split_with_sentinel}')
                cst = ''
                cet = sequence[token_index[0]]
                fst = sequence[token_index[0] + 1]
                fet = ''
                condition_sequence = ''
                focus_sequence = ''
                for cond_start_token in self.tokens_dictionary['cond_start_tokens']:
                    condition_splits = condition_split_with_sentinel.split(cond_start_token)
                    if len(condition_splits) == 2:
                        print(f'condition_splits:{condition_splits}')
                        cst = cond_start_token
                        condition_sequence = condition_splits[1]
                        break
                unused_cond_tokens = max_cond_sequence_length - len(condition_sequence)
                print("unused_cond_tokens:",unused_cond_tokens)
                pad_cond = []
                for i in  range(unused_cond_tokens):
                    pad_cond.append(self.pad_token_id )
                condition_sequence = cst + condition_sequence[:max_cond_sequence_length] + cet + pad_cond

                for focus_start_token in self.tokens_dictionary['focus_start_tokens']:
                    focus_splits = focus_split_with_sentinel.split(focus_start_token)
                    if len(focus_splits) == 2:
                        print(f'focus:{focus_splits}')
                        fst = focus_start_token
                        focus_sequence = focus_splits[1]
                        break
                unused_focus_tokens = max_focus_sequence_length - len(focus_sequence)
                print("unused_focus_tokens:",unused_focus_tokens)
                pad_cond = []
                for i in  range(unused_focus_tokens):
                    pad_cond.append(self.pad_token_id)
                focus_sequence = fst + focus_sequence[:max_focus_sequence_length] + fet + pad_cond
                print(f'condition_sequence:{condition_sequence} focus_sequence:{focus_sequence}')
                new_sequence = condition_sequence +sentinel_token + focus_sequence
                print(f'sequence:{sequence} new_sequence:{new_sequence}')
                output = new_sequence
                # output = self.trie.tokenize(new_sequence)
            else:
                sequence = old_sequence
                output = self.trie.tokenize(sequence)
        else:
            sequence = old_sequence
            output = self.trie.tokenize(sequence)


        if return_tensor:
            output = torch.tensor(output, dtype=torch.long)
        
        if max_sequence_length is not None:
            output = output[:max_sequence_length]
            

        return output

    def batch_encode_pair(
        self,
        sequences: List[str],
        return_tensors: bool = False,
        max_cond_sequence_length: Optional[int] = None,
        max_focus_sequence_length: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ) -> List[List[int]]: #specific encode function for sequences from PairedDataset
        def get_index(sequence,token):
            
            try:
                index = sequence.index(token)
            except Exception as e:
                # print('get_index_exception:')
                # print(e)
                index = -1
            # print('get_index(sequence,token)')
            # print(sequence,token,index)
            return index
        output = []
        token_index_list = []
        if max_sequence_length is None and return_tensors:
            max_sequence_length = max([len(sequence) for sequence in sequences])
            
        for sequence in sequences:
            output.append(self.encode_pair(sequence, return_tensors))
            token_index = []
            for token in self.ids_to_sentinel_tokens:
                index = get_index(sequence[:max_sequence_length],token)
                if index > -1:
                    token_index.append(index)
            token_index_list.append(token_index)
            # token_index_list.append({token:get_index(sequence[:max_sequence_length],token) for token in self.ids_to_tokens})
        if return_tensors:
            tensor_out = torch.full((len(output), max_sequence_length), self.pad_token_id)
            for i, sequence in enumerate(output):
                tensor_out[i, :len(sequence)] = sequence
            output = tensor_out
        return output,token_index_list

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
            "cond_start_tokens": ["<bc-30>","<bc-50>","<bc-90>","<bc-str>"],
            "tokens": ["L", "A", "G", "V", 
            "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", 
            "F", "Y", "M", "H", "W", "C", "B", "U", "Z", "O"],
            "cond_end_tokens":  ["<eoc>"],
            "special_tokens": ["<pad>","<mask>"],
            "focus_start_tokens" : ["<bf-lr>","<bf-rl>","<bf-fim-lr>","<bf-fim-l>","<bf-fim-r>","<bf-fim-m>",
                                    "<bf-mo-l>","<bf-mo-r>","<bf-mo-n>"],
            "focus_end_tokens": ["<eof>"],
            "end_tokens": ["<eos>"],
        }
        self.tokens = [token for key,tokens in self.tokens_dictionary.items() for token in tokens]
        self.sentinel_tokens = [token  for key in ["cond_start_tokens","cond_end_tokens","focus_start_tokens","focus_end_tokens"] for token in self.tokens_dictionary[key]] 
        self.sentinel_start_tokens = [token  for key in ["cond_end_tokens","end_tokens"] for token in self.tokens_dictionary[key]] 
        self.sentinel_end_tokens = [token  for key in ["cond_end_tokens","end_tokens"] for token in self.tokens_dictionary[key]] 
        #the assumption each paired sequence example will have a single cond_end_token and no end_token
        # and a normal single sequence example to have only a single end_token and neither of cond_end or focus_end tokens
        print('###setinel_tokens###')
        print(self.sentinel_tokens)
        super().__init__(self.tokens,self.sentinel_end_tokens,self.tokens_dictionary)


    
    