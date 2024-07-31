import torch
import numpy as np
from typing import List, Union, Optional
from rust_trie import Trie 
from protein_lm.modeling.utils.attention_mask import create_attention_mask


class Tokenizer:
    def __init__(self, tokens: List[str],tokens_dictionary:Optional[dict] = None, unk_token_id: Optional[int] = None,sequence_mask_fraction: Optional[dict]= None,mask_type:Optional[str] = "random"):
        self.ids_to_tokens = tokens
        self.tokens_dictionary = tokens_dictionary
        self.sequence_mask_fraction = sequence_mask_fraction
        self.mask_type = mask_type
        # self.trie = Trie(unk_token_id)
        # for token in tokens:
            # self.trie.add(token)
        # If unk_token_id is not provided, add <unk> to the end of the tokens list
        if unk_token_id is None:
            self.ids_to_tokens += ["<unk>"]
            self.unk_token = "<unk>"
            self.unk_token_id = self.ids_to_tokens.index("<unk>")
        else:
            self.unk_token = self.ids_to_tokens[unk_token_id]
            self.unk_token_id = unk_token_id
        self.pad_token_id = self.ids_to_tokens.index("<pad>")
        self.mask_token_id = self.ids_to_tokens.index("<mask>")
        self.eoc_token = self.ids_to_tokens.index("<eoc>")
        self.eof_token = self.ids_to_tokens.index("<eof>")
        # self.prefix_tokens = [self.ids_to_tokens.index(f"<bf-{cluster}-{decoding_order}-{position}>")  for cluster in [30,50,90,100] for decoding_order in ['fim','mo'] for position in ['l'] ]
        # self.middle_tokens = [self.ids_to_tokens.index(f"<bf-{cluster}-{decoding_order}-{position}>")  for cluster in [30,50,90,100] for decoding_order in ['fim','mo'] for position in ['m'] ]
        # self.suffix_tokens = [self.ids_to_tokens.index(f"<bf-{cluster}-{decoding_order}-{position}>")  for cluster in [30,50,90,100] for decoding_order in ['fim','mo'] for position in ['r'] ]
        self.encoder = {token:self.ids_to_tokens.index(token) for token in tokens}
        self.decoder = {self.ids_to_tokens.index(token):token for token in tokens}
        self.vocab_size = len(self.ids_to_tokens)


    def __call__(self,sequences: Union[str, List] ,focus_sequences: Optional[Union[str, List]],cond_clusters: Optional[Union[str, List]],focus_clusters: Optional[Union[str, List]],decoding_orders: Optional[Union[str, List]], multisequence:Optional[bool]= False, *args, **kwargs):
        if multisequence:
            check1 = all([isinstance(sequences, str), isinstance(focus_sequences, str), isinstance(cond_clusters, str), isinstance(focus_clusters, str), isinstance(decoding_orders, str)])
            check2 = all([isinstance(sequences, list), isinstance(focus_sequences, list), isinstance(cond_clusters, list), isinstance(focus_clusters, list), isinstance(decoding_orders, list)])
            if check1:
                return self.encode_multisequence(sequences,focus_sequences,cond_clusters,focus_clusters,decoding_orders, *args, **kwargs)
            elif check2:
                #sequences arguement will be considered as the condition sequences
                c,f,cc,fc,d = len(sequences),len(focus_sequences),len(cond_clusters),len(focus_clusters),len(decoding_orders)
                if c == f and f > 0 and cc == fc and fc > 0: #all lengths should be non zero and equal to one another
                    return self.batch_encode_multisequence(sequences,focus_sequences,cond_clusters,focus_clusters,decoding_orders, *args, **kwargs)
                else:
                    raise Exception(f"ERROR: Length mismatch between condition sequences (length={c}) focus sequences (length={f}) sequences,condition clusters (length={cc}) focus clusters (length={fc}) clusters and decoding_orders (length={d})")
            else:
                raise Exception(f"ERROR: Some inputs are lists and some are string. Ensure all passed arguments are either all string or all lists (of equal lengths)")
        else:
            if isinstance(sequences, str):
                return self.encode(sequences, *args, **kwargs)
            else:
                return self.batch_encode(sequences, *args, **kwargs)
    
    def compute_sentinel_indices(self,tokenized_sequence: Union[str, list],has_cond_sequence: bool ,has_focus_sequence: bool, add_special_tokens:bool, return_tensor:bool,max_cond_sequence_length:int,max_focus_sequence_length:int,max_sequence_length:int):       
        print(f'INFO:compute_sentinel_indices->tokenized_sequence:{tokenized_sequence}')
        # print(f'INFO: eoc_token:{eoc_token} eof_token:{eof_token}')
        indices = [0,0] # index of padding_start,padding_end for condition
        if add_special_tokens:
            max_cond_sequence_length += 2 #we decremented it by to incorporate <boc> and <eoc>
        if return_tensor:
            if has_cond_sequence:
                try:
                    print(f'tensor:tokenized_sequence:{tokenized_sequence} self.eoc_token:{self.eoc_token}')
                    indices[0] = (tokenized_sequence == self.eoc_token).nonzero().item() + 1
                    indices[1] = max_cond_sequence_length 
                except Exception as e:
                    print(f'EXCEPTION:<eoc> tensor indices:{indices} {e}')
            
        else:
            if has_cond_sequence:
                try:
                    print(f'list:tokenized_sequence:{tokenized_sequence} self.eoc_token:{self.eoc_token}')
                    indices[0] = tokenized_sequence.index(self.eoc_token) + 1
                    indices[1] = max_cond_sequence_length 
                except Exception as e:
                    print(f'EXCEPTION:<eoc> list indices:{indices} {e}')

        print(f'INFO: sentinel_indices:{indices}')
        
        return indices

    def batch_compute_sentinel_indices(self,tokenized_sequences: Union[str, list],cond_sequences: Union[str, list] ,focus_sequences: Optional[Union[str, list]], add_special_tokens:bool, return_tensors:bool,max_cond_sequence_length:int,max_focus_sequence_length:int,max_sequence_length:int):
        print(f' compute_sentinel_indices({tokenized_sequences}:,{cond_sequences:},{focus_sequences}')
        if return_tensors:
            check1 = isinstance(tokenized_sequences,torch.Tensor)
            check2 = isinstance(cond_sequences,torch.Tensor)
            check3 = isinstance(focus_sequences,torch.Tensor)
            if check1 and check2 and check3:
                check1 = all([token.dtype == torch.long for token in tokenized_sequences])
                check2 = all([token.dtype == torch.long for token in cond_sequences])
                check3 = all([token.dtype == torch.long for token in focus_sequences])
        else:
            check1 = all([isinstance(token, int) for token in tokenized_sequences])
            check2 = all([isinstance(token, int) for token in cond_sequences])
            check3 = all([isinstance(token, int) for token in focus_sequences])
        
        print(f'check1:{check1} check2:{check2} check3:{check3}')
        if check1 and check2 and check3: # when initializing the tokenizer, it is already checked whether each of tokenized/cond/focus have same num_samples
            tokenized_sequences = [tokenized_sequences]
            cond_sequences = [cond_sequences]
            focus_sequences = [focus_sequences]

        if isinstance(tokenized_sequences,str) and isinstance(cond_sequences,str) and isinstance(focus_sequences,str) : # when initializing the tokenizer, it is already checked whether each of tokenized/cond/focus have same num_samples
            tokenized_sequences = [tokenized_sequences]
            cond_sequences = [cond_sequences]
            focus_sequences = [focus_sequences] 
        
                  
        sentinel_indices = []
        print(f'compute_sentinel_indices:{tokenized_sequences}')
        for i in range(len(cond_sequences)):
            print(f'INFO:tokenized_sequence:{tokenized_sequence_decoded} from condition:{cond_sequences[i]} focus:{focus_sequences[i]}')
            # print(f'INFO: eoc_token:{eoc_token} eof_token:{eof_token}')
            indices = [0,0] # index of padding_start,padding_end for condition,focus_padding_start
            if return_tensors:
                if len(cond_sequences[i]):
                    try:
                        indices[0] = (tokenized_sequences[i] == self.eoc_token).nonzero().item() + 1
                        indices[1] = max_cond_sequence_length 
                    except Exception as e:
                        print(f'EXCEPTION:<eoc> tensor indices:{indices} {e}')
            else:
                if len(cond_sequences[i]):
                    try:
                        indices[0] = tokenized_sequences[i].index(self.eoc_token) + 1
                        indices[1] = max_cond_sequence_length 
                    except Exception as e:
                        print(f'EXCEPTION:<eoc> list indices:{indices} {e}')
                
            print(f'INFO: sentinel_indices:{indices}')
            sentinel_indices.append(indices)
        return sentinel_indices

    def mask_sequence(self,tokenized_sequence:list,sentinel_index:list,cluster:str,return_tensor:bool=True):
        cond_sequence,focus_sequence = tokenized_sequence[:sentinel_index[0]],tokenized_sequence[sentinel_index[1]:]
        seq_len,cond_seq_len,focus_seq_len = len(tokenized_sequence),len(cond_sequence),len(focus_sequence)
        cond_pad_idx = [i for i in range(sentinel_index[0],sentinel_index[1])]
        focus_pad_idx = [i for i in range(sentinel_index[1],seq_len) if tokenized_sequence[i] == self.pad_token_id]
        focus_idx = [i for i in range(sentinel_index[1],seq_len) if tokenized_sequence[i] != self.pad_token_id]
        pad_idx = cond_pad_idx + focus_pad_idx
        if cond_seq_len:
            masked_sequence,mask = create_attention_mask(tokenized_sequence = tokenized_sequence, padding_idx = pad_idx,focus_idx=focus_idx,span_length = 5,sequence_mask_fraction= self.sequence_mask_fraction['condition'][cluster],mask_type = self.mask_type,mask_id=self.mask_token_id,return_tensor=return_tensor)
        else:
            masked_sequence = tokenized_sequence
            mask = torch.ones(seq_len)
            mask[pad_idx] = 1
        return masked_sequence,mask
        




        
    def get_encoder_value(self,tokens:str):
        if not tokens:
            return []
        print(f'get_encoder_value:tokens:{tokens}')    
        token_ids = []
        parent_node = "<"
        leaf_node = ">"
        subtoken = ""
        max_subtoken_length = 14 # the maximum a subtoken length can be, currently it is the maximum tag length in the vocabulary e.g <bf-100-fim-l>
        for token in tokens:
            token = str(token)
            if token in self.ids_to_tokens: #will match for AA tokens,however this could be an expensive operation for large vocabulary
                token_ids.append(self.encoder[token])
                print(f'get_encoder_value:token:{token} -> {token_ids[-1]}')    
            else:
                
                if token == parent_node: #will check for tokens starting with a tag
                    start_subtoken = True
                if start_subtoken: # will start creating a word starting from parent_node till leaf_node e.g a token like <bc-50>
                    subtoken = subtoken + token
                    if len(subtoken) > max_subtoken_length:
                        if subtoken in self.ids_to_tokens:
                            #the subtoken exists in the vocabulary
                            token_ids.append(self.encoder[subtoken])
                        else:
                            #the subtoken beginning from parent to child node is not in the vocabulary so unknown token
                            token_ids.append(self.unk_token_id)


                    if token == leaf_node: #we are assuming we will encounter a leaf node/token
                        if subtoken in self.ids_to_tokens:
                            #the subtoken exists in the vocabulary
                             
                            token_ids.append(self.encoder[subtoken])
                            if len(token_ids):
                                print(f'get_encoder_value:subtoken:{subtoken} -> {token_ids[-1]}') 
                        else:
                            #the subtoken beginning from parent to child node is not in the vocabulary so unknown token
                            token_ids.append(self.unk_token_id)
                            if len(token_ids):
                                print(f'get_encoder_value:subtoken:{subtoken} -> {token_ids[-1]}') 
                        
                        subtoken = ""
                        start_subtoken = False
                else: #will apply to tokens not AA tokens and tag tokens,therefore unknown token
                    token_ids.append(self.unk_token_id)
                  
        return token_ids
    

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
        output = self.get_encoder_value(sequence)
        # output = self.trie.tokenize(sequence)
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
        # print(f"""INFO: encode_multisequence:  
        # cond_sequence:{cond_sequence},
        # focus_sequence:{focus_sequence}, 
        # cond_sequence_cluster:{cond_sequence_cluster},
        # focus_sequence_cluster:{focus_sequence_cluster},
        # decoding_order:{decoding_order},
        # add_special_tokens:{add_special_tokens}
        # return_tensor:{return_tensor}
        # max_cond_sequence_length:{max_cond_sequence_length}
        # max_focus_sequence_length:{max_focus_sequence_length}
        # max_sequence_length:{max_sequence_length}""")
        # c,f = len(cond_sequence),len(focus_sequence)
        # if c and not len(cond_sequence_cluster):
        #     print(f'WARNING CLUSTER: condition sequence provide but not cluster')
        # print(f'INFO: len(condition_sequence): {c} len(focus_sequence): {f}')
        sequence = ""
        if max_cond_sequence_length is not None:
            if add_special_tokens:
                #ensure consideration of sentinel tokens
                max_cond_sequence_length -= 2 #4 tokens:<boc><eoc>
            cond_sequence = cond_sequence[:max_cond_sequence_length]
        
        if max_focus_sequence_length is not None:
            if add_special_tokens:
                #ensure consideration of sentinel tokens
                if decoding_order in ['mo','fim']:
                    max_focus_sequence_length -= 4 #6 tokens:<bf-mo-l>,<bf-mo-m>,<bf-mo-r>,<eof>
                else:
                    max_focus_sequence_length -= 2 # 4 tokens:<bof><eof>
                    
            focus_sequence = focus_sequence[:max_focus_sequence_length]
        # print(f'INFO:cond_sequence[:max_cond_sequence_length]->cond_sequence[:{max_cond_sequence_length}]={cond_sequence}')
        # print(f'INFO:focus_sequence[:max_focus_sequence_length]->focus_sequence[:{max_focus_sequence_length}]={focus_sequence}')
        c,f = len(cond_sequence),len(focus_sequence)
        if max_cond_sequence_length < c or max_focus_sequence_length < f:
            print(f'WARNING: one of condition or focus sequence has been specified a maximum length smaller than actual sequence length')
        if f == 0:
            #pad the whole sequence till model capacity since no focus sequence
            cond_pad_length = max_sequence_length - c #unused tokens
        else:
            #pad the sequence till condition sequence capacity since we have focus sequence
            cond_pad_length = max_cond_sequence_length - c #unused tokens
        cond_pad_sequence = "".join(["<pad>" for i in range(cond_pad_length)])
        
        if add_special_tokens:
            
            cluster_checks = ['30','50','70','90','100','afdb']
            decoding_order_checks = ['lr']
            cond_modified_sequence = ''
            focus_modified_sequence = ''
            for check in cluster_checks:
                print(f'INFO:cluster_check:{check} cond_sequence_cluster:{cond_sequence_cluster} focus_sequence_cluster:{focus_sequence_cluster} decoding_order:{decoding_order} cond_sequence:{cond_sequence} focus_sequence:{focus_sequence}')

                if c > 0 : # condition sequence exists
                    if check in cond_sequence_cluster:
                        cond_start_token = [t for t in self.tokens_dictionary['cond_start_tokens'] if check in t][0]
                        cond_end_token = self.tokens_dictionary['cond_end_tokens'][0]
                        cond_modified_sequence = cond_start_token + cond_sequence  + cond_end_token + cond_pad_sequence
                        # print(f'INFO:condition_check:{check} decoding_order:{decoding_order} cond_modified_sequence:{cond_modified_sequence}')
                if f > 0: #focus sequence exists
                    if check in focus_sequence_cluster:
                        focus_end_token = self.tokens_dictionary['focus_end_tokens'][0]
                        decoding_order_to_sequence = {"lr": focus_sequence,
                                                      "rl":focus_sequence[::-1],
                                                      "mo":self.encode_mo(focus_sequence),
                                                      "fim":self.encode_fim(focus_sequence)
                        }
                        if decoding_order in ['mo','fim']:
                            focus_pad_length = max_focus_sequence_length - f   #unused tokens
                            focus_pad_sequence =  "".join(["<pad>" for i in range(focus_pad_length)])
                            
                            prefix_start_token = [t for t in self.tokens_dictionary['focus_start_tokens'] if check in t and  decoding_order +'-l' in t ][0]
                            middle_start_token = [t for t in self.tokens_dictionary['focus_start_tokens'] if check in t and  decoding_order +'-m' in t ][0]
                            suffix_start_token = [t for t in self.tokens_dictionary['focus_start_tokens'] if check in t and  decoding_order +'-r' in t][0]
                            # print(f'INFO:prefix_start_token:{prefix_start_token} middle_start_token:{middle_start_token} suffix_start_token:{suffix_start_token}')
                            prefix_sequence,middle_sequence,suffix_sequence = decoding_order_to_sequence[decoding_order]
                            focus_modified_sequence =  prefix_start_token + prefix_sequence + middle_start_token + middle_sequence + suffix_start_token + suffix_sequence + focus_end_token + focus_pad_sequence
                            # print(f'INFO:focus_check:{check} decoding_order:{decoding_order} focus_modified_sequence:{focus_modified_sequence}')
                        else:
                            if c == 0:
                                focus_pad_length = max_sequence_length - f  # unused tokens
                            else:
                                #pad the sequence till focus sequence capacity since we have condition sequence
                                focus_pad_length = max_focus_sequence_length - f  # unused tokens
                            focus_pad_sequence =  "".join(["<pad>" for i in range(focus_pad_length)])
                            # print(f'INFO:check:{check} decoding_order:{decoding_order}')
                            focus_start_token = [t for t in self.tokens_dictionary['focus_start_tokens'] if check in t and  decoding_order  in t][0]
                            focus_sequence = decoding_order_to_sequence[decoding_order]
                            focus_modified_sequence = focus_start_token + focus_sequence + focus_end_token + focus_pad_sequence
                            # print(f'INFO:focus_check:{check} decoding_order:{decoding_order} focus_modified_sequence:{focus_modified_sequence}') 
            # if not len(cond_modified_sequence) and not len(focus_modified_sequence):
                # print(f'WARNING:both cond_modified_sequence:{cond_modified_sequence} focus_modified_sequence:{focus_modified_sequence} are empty')


            sequence = cond_modified_sequence + focus_modified_sequence
            # print(f'INFO:cond_modified_sequence:{cond_modified_sequence} focus_modified_sequence:{focus_modified_sequence} sequence:{sequence}')
                
        else:
            #special tokens except padding are already added
            sequence = cond_sequence + cond_pad_sequence + focus_sequence + focus_pad_sequence

        # print(f'INFO: tokenizing sequence:{sequence}')
        # output = self.trie.tokenize(sequence)
        print(f'INFO: get_encoder_value-> Tokenizing sequence: {sequence}')
        output = self.get_encoder_value(sequence)
        print(f'max_sequence_length:{max_sequence_length}')
        output = output[:max_sequence_length]
        # if self.eoc_token not in output and len(cond_sequence):
        #     try:
        #         cond_pad_start = output.index(self.pad_token_id)
        #     except:
        #         cond_pad_start = max_cond_sequence_length 
            
        #     output = output[:cond_pad_start - 1] + [self.eoc_token] + output[cond_pad_start:]
        #     print(f'INFO: Added eoc_token{self.eoc_token} to output:{output}')
        # if self.eof_token not in output and len(focus_sequence):
        #     output =  output[:max_sequence_length - 1] + [self.eof_token]
        #     print(f'INFO: Added eof_token{self.eof_token} to output:{output}')
        print(f'INFO: tokenizing output:{output} len:{len(output)}')
        o = len(output)
        
        if o < max_sequence_length:
            difference = max_sequence_length - o
            # print(f'WARNING:output:{output} len(output):{o} < max_sequence_length:{max_sequence_length}')      
            # print(f'WARNING:filling difference:{difference} with padding sequence')
            padding_sequence = [self.pad_token_id for i in range(difference)]
            output.extend(padding_sequence)
            # new_sequence = self.decode(output)
            # print(f'INFO: New Filled sequence:{new_sequence}')
        print(f'INFO: tokenized ouput:{output} len:{len(output)}')
        # if len(output) > max_sequence_length:
            # print(f'WARNING: tokenized length {len(output)} has exceed maximum length:{max_sequence_length}')
        
        if return_tensor:
            output = torch.tensor(output, dtype=torch.long) # need to work to make sure len(output) < max_sequence_length
        
        has_cond_sequence = len(cond_sequence[:max_cond_sequence_length]) > 0
        has_focus_sequence = len(focus_sequence[:max_focus_sequence_length]) > 0
        sentinel_index = self.compute_sentinel_indices(output,has_cond_sequence,has_focus_sequence, add_special_tokens,return_tensor,max_cond_sequence_length,max_focus_sequence_length,max_sequence_length)
        
        masked_output,attention_mask = self.mask_sequence(output,sentinel_index,cond_sequence_cluster,return_tensor)
        print(f'output:{output} attention_mask:{attention_mask}')
        return masked_output,output,sentinel_index,attention_mask

    def batch_encode_multisequence(
        self,
        cond_sequences: List[str],
        focus_sequences: List[str],
        cond_sequence_clusters: List[str],
        focus_sequence_clusters: List[str],
        decoding_orders: List[str],
        add_special_tokens: bool = False,
        return_tensors: bool = False,
        max_cond_sequence_length: Optional[int] = None,
        max_focus_sequence_length: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ) -> List[List[int]]:
    #     print(f"""
    #     batch_encode_multisequence(
        
    #     cond_sequences: {cond_sequences},
    #     focus_sequences: {focus_sequences},
    #     cond_sequence_clusters: {cond_sequence_clusters},
    #     focus_sequence_clusters: {focus_sequence_clusters},
    #     decoding_orders: {decoding_orders},
    #     add_special_tokens: bool = {add_special_tokens},
    #     return_tensors: bool = {return_tensors},
    #     max_cond_sequence_length: Optional[int] = {max_cond_sequence_length},
    #     max_focus_sequence_length: Optional[int] = {max_focus_sequence_length},
    #     max_sequence_length: Optional[int] = {max_sequence_length},
    # )
    #     """)
        
        if max_cond_sequence_length is None and return_tensors:
            max_cond_sequence_length = max([len(sequence) for sequence in cond_sequences])
            if add_special_tokens:
                max_cond_sequence_length += 2 #4 tokens:<boc><eoc>
        if max_cond_sequence_length is not None:
            cond_sequences = [
                sequence[:(max_cond_sequence_length - 2) if add_special_tokens else max_cond_sequence_length] 
                for sequence in cond_sequences
            ]
        
        if max_focus_sequence_length is None and return_tensors:
            max_focus_sequence_length = max([len(sequence) for sequence in focus_sequences])
            if add_special_tokens:
                #4 maximum sentinel tokens possible:<bf-mo-l>,<bf-mo-m>,<bf-mo-r>,<eof> for Middle out or FIM, it is posisble there are only two for LR and RL
                #but we are considering the maximum case below
                max_focus_sequence_length += 4
                
                
        if max_focus_sequence_length is not None:
            for i,sequence in enumerate(focus_sequences):
                if add_special_tokens:
                    if decoding_orders[i] in ['mo','fim']:
                        focus_sequences[i] = sequence[:(max_focus_sequence_length - 4)] # 4 tokens: <bf-mo-l>,<bf-mo-m>,<bf-mo-r>,<eof>,
                    else:
                        focus_sequences[i] = sequence[:(max_focus_sequence_length - 2)] #2 tokens: <bof><eof>
                else:
                    focus_sequences[i] = sequence[:(max_focus_sequence_length)]

                        
            
            
       
        eoc_token = self.ids_to_tokens.index("<eoc>")
        eof_token = self.ids_to_tokens.index("<eof>")
        prefix_tokens = [self.ids_to_tokens.index(f"<bf-{cluster}-{decoding_order}-{position}>")  for cluster in [30,50,70,90,100] for decoding_order in ['fim','mo'] for position in ['l'] ]
        middle_tokens = [self.ids_to_tokens.index(f"<bf-{cluster}-{decoding_order}-{position}>")  for cluster in [30,50,70,90,100] for decoding_order in ['fim','mo'] for position in ['m'] ]
        suffix_tokens = [self.ids_to_tokens.index(f"<bf-{cluster}-{decoding_order}-{position}>")  for cluster in [30,50,70,90,100] for decoding_order in ['fim','mo'] for position in ['r'] ]
        sentinel_indices = []
        output = []
        masked_output = []
        attention_masks = []
        for i in range(len(cond_sequences)):
            # print(f"""
            # INFO:tokenization #{i}:cond_sequences[i]={cond_sequences[i]},focus_sequences[i]={focus_sequences[i]},
            # cond_sequence_clusters[i]={cond_sequence_clusters[i]},focus_sequence_clusters[i]={focus_sequence_clusters[i]},
            # decoding_orders[i]={decoding_orders[i]}, add_special_tokens={add_special_tokens}, 
            # return_tensors={return_tensors},max_cond_sequence_length={max_cond_sequence_length},max_focus_sequence_length={max_focus_sequence_length},
            # max_sequence_length={max_sequence_length}
            # """)
            
            

            masked_tokenized_sequence,tokenized_sequence,sentinel_index,attention_mask = self.encode_multisequence(cond_sequences[i],focus_sequences[i],cond_sequence_clusters[i],focus_sequence_clusters[i],decoding_orders[i], add_special_tokens, return_tensors,max_cond_sequence_length,max_focus_sequence_length,max_sequence_length)
            tokenized_sequence_decoded = self.decode(tokenized_sequence)
            sentinel_indices.append(sentinel_index)
            print(f'INFO:tokenized_sequence:{tokenized_sequence_decoded} from condition:{cond_sequences[i]} focus:{focus_sequences[i]} sentinel_index:{sentinel_index}')
            output.append(tokenized_sequence)
            masked_output.append(masked_tokenized_sequence)
            atention_masks.append(attention_mask)             
            
        # sentinel_indices = self.batch_compute_sentinel_indices(output,cond_sequences,focus_sequences, add_special_tokens,return_tensors,max_cond_sequence_length,max_focus_sequence_length,max_sequence_length)
        print(f'sentinel_indices:{sentinel_indices}')
        
        if return_tensors:
            tensor_out = torch.full((len(output),max_sequence_length), self.pad_token_id)
            tensor_masked_out = torch.full((len(masked_output),max_sequence_length), self.pad_token_id)
            attention_masks = torch.tensor(attenion_masks)
            for i, sequence in enumerate(output):
                tensor_out[i, :len(sequence)] = sequence
                tensor_masked_out[i, :len(masked_output[i])] = masked_output[i]
            output = tensor_out
            masked_output = tensor_masked_out
        return masked_output ,output,sentinel_indices,attention_masks

    def encode_fim(self, sequence:str) -> str:
        #expect a pure sequence without padding/sentinel tokens
        s = len(sequence)
        
        print(f'INFO:FIM:sequence:{sequence} sequence_length:{s}')
        prefix_sequence = ''
        middle_sequence = ''
        suffix_sequence = ''
        if s == 0:
            return '','',''
        if s == 1:
            return sequence,'',''
        if s == 2:
            return sequence[0],sequence[1],''
        if s == 3:
            return sequence[0],sequence[1],sequence[2]
        iteration = 0
        while not prefix_sequence and not middle_sequence and not suffix_sequence: #ensure no part is empty
            iteration +=1
            print(f'INFO:FIM:iteration # {iteration} for FIM')
            feasible_middle_indices = list(range(int(0.2 * s),int(0.8 * s))) #ensure the start of middle index will be atleast between the middle 60% (after the first 20% tokens and before last 20% tokens ) e.g <20%>|<60%>|<20%>
            feasible_middle_lengths = list(range(max(int(0.02 * s), 1), int(0.5 * s))) #length can be only betwee 2% of sequence_length or 1 and 50% of sequence 
            print(f'INFO:FIM:feasible_middle_indices:{feasible_middle_indices} feasible_middle_lengths:{feasible_middle_lengths}')
            middle_index =  np.random.choice(feasible_middle_indices) # randomly pick a middle index
            middle_length =  np.random.choice(feasible_middle_lengths) # randomly pick length of middle region
            print(f'INFO:FIM:middle_index:{middle_index} middle_length:{middle_length}')
            prefix_sequence = sequence[:middle_index]
            middle_sequence = sequence[middle_index:middle_index + middle_length]
            suffix_sequence = sequence[middle_index + middle_length:]
        
        print(f'INFO:FIM:prefix_sequence:{prefix_sequence} middle_sequence:{middle_sequence} suffix_sequence:{suffix_sequence}')

        return prefix_sequence,middle_sequence,suffix_sequence

    def encode_mo(self, sequence:str) -> str:
        #expect a pure sequence without padding/sentinel tokens
        s = len(sequence)
        print(f'INFO:MO:sequence:{sequence} sequence_length:{s}')
        prefix_sequence = ''
        middle_sequence = ''
        suffix_sequence = ''
        iteration  = 0
        if s == 0:
            return '','',''
        if s == 1:
            return sequence,'',''
        if s == 2:
            return sequence[0],sequence[1],''
        if s == 3:
            return sequence[0],sequence[1],sequence[2]
        while not prefix_sequence and not middle_sequence and not suffix_sequence:  #ensure no part is empty
        
            iteration +=1
            print(f'INFO:MOiteration # {iteration} for MO')
            feasible_middle_indices = list(range(int(0.2 * s),int(0.8 * s))) #ensure the start of middle index will be atleast between the middle 60% (after the first 20% tokens and before last 20% tokens ) e.g <20%>|<60%>|<20%>
            #considering a slightly longer middle length here since we will be predicting both prefix and suffix regions
            feasible_middle_lengths = list(range(max(int(0.2 * s), 1), int(0.5 * s))) #length can be only betwee 20% of sequence_length or 1 and 50% of sequence 
            print(f'INFO:MO:feasible_middle_indices:{feasible_middle_indices} feasible_middle_lengths:{feasible_middle_lengths}')
            middle_index =  np.random.choice(feasible_middle_indices) # randomly pick a middle index
            middle_length =  np.random.choice(feasible_middle_lengths) # randomly pick length of middle region
            print(f'INFO:MO:middle_index:{middle_index} middle_length:{middle_length}')
            prefix_sequence = sequence[:middle_index]
            middle_sequence = sequence[middle_index:middle_index + middle_length]
            suffix_sequence = sequence[middle_index + middle_length:]
        print(f'INFO MO::prefix_sequence:{prefix_sequence} middle_sequence:{middle_sequence} suffix_sequence:{suffix_sequence}')
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
    def __init__(self,sequence_mask_fraction={},mask_type = "random"):
        # For our own tokenizers, we don't need to explicitly add the <unk> token
        # because it gets added as the last token in the tokens list
        # I've also removed X so that it gets translated to <unk>
        self.tokens_dictionary = {
            "special_tokens": ["<pad>","<mask>"],
            "start_tokens": ["<cls>"],
            "cond_start_tokens": ["<bc-30>","<bc-50>","<bc-70>","<bc-90>","<bc-100>","<bc-afdb>"],
            "tokens": ["L", "A", "G", "V", 
            "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", 
            "F", "Y", "M", "H", "W", "C", "B", "U", "Z", "O"],
            "cond_end_tokens":  ["<eoc>"],
            "focus_start_tokens" : ["<bf-30-lr>","<bf-30-rl>",
                                    "<bf-50-lr>","<bf-50-rl>",
                                    "<bf-70-lr>","<bf-70-rl>",
                                    "<bf-90-lr>","<bf-90-rl>",
                                    "<bf-100-lr>","<bf-100-rl>",
                                    "<bf-afdb-lr>","<bf-afdb-rl>",
                                    ],
            "focus_end_tokens": ["<eof>"],
            "end_tokens": ["<eos>"],
        }
        self.tokens = [token for key,tokens in self.tokens_dictionary.items() for token in tokens]
        self.sequence_mask_fraction = sequence_mask_fraction
        self.mask_type = mask_type
        self.pad_token_id = self.tokens.index("<pad>")
        self.mask_token_id = self.tokens.index("<mask>")
        super().__init__(self.tokens,tokens_dictionary=self.tokens_dictionary,sequence_mask_fraction = self.sequence_mask_fraction,mask_type=self.mask_type)