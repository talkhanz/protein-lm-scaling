import torch
import pandas as pd
import os
import numpy as np
from typing import Union
import csv


class MultiSequenceIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
            self, 
            dataset_path: str = "", 
            cluster_table_path: str = "",
            index_column:str="index",
            focus_sequence_column:str = "U100",
            condition_sequence_columns:dict = {'uniref30': 'UniRef30_clust_str','uniref50': 'UniRef50_clust_str','uniref70': 'UniRef70_clust_str','uniref90':'UniRef90_clust_str','afdb': 'Alphafold_clust_str'}, 
            cond_sequence_length:int = 5,
            focus_sequence_length:int = 5,
            max_sequence_length:int = 10,
            pair_prob:float = 0.7, #probably a sample will have a pair of sequences
            cluster_prob:dict = {'uniref30': 0.2,'uniref50':0.2,'uniref70': 0.2,'uniref90': 0.2,'struct': 0.2}, #if a pair is selected, probably of condition sequence per cluster
            decoding_order_prob:dict = {'lr': 1,'rl': 0, 'fim': 0,'mo': 0}, #independant of paired sequence or singleton
            separator_token:str=",", #the token which separates the list of sequences in the condition sequence columns
            tokenization_strategy:str="precompute",
            tokenizer = None,
            seed: int = 42,     
        ) -> None:
       
        
        self.dataset_path = dataset_path
        self.probabilities_dict = {"pair": pair_prob,'cluster': cluster_prob,'decoding_order':decoding_order_prob}
        self.columns = {"condition":condition_sequence_columns,"focus":focus_sequence_column,"index":index_column}
        self.max_sequence_length_dict = {'condition': cond_sequence_length,'focus': focus_sequence_length,'total': max_sequence_length}
        self.separator_token = separator_token
        self.cluster_table = pd.read_csv(cluster_table_path) if cluster_table_path else None
        self.tokenization_strategy = tokenization_strategy
        self.tokenizer = tokenizer
        np.random.seed(seed)
    


    def __iter__(self):
        col2id =  {self.columns[key]:0 for key in ['focus','index']}
        col2id.update({value:0 for key,value in self.columns['condition'].items()})
        with open(self.dataset_path, mode='r') as file:
            reader = csv.reader(file)
            for idx,row in enumerate(reader):
                if idx == 0:
                    for i,item in enumerate(row):
                        col2id[item] = i
                    
                
                if idx > 0: # start yielding samples from second row of file,assume first line has headers
                    index = row[col2id[self.columns['index']]]
                    is_pair = np.random.uniform() < self.probabilities_dict['pair']
                    decoding_order = np.random.choice(list(self.probabilities_dict['decoding_order'].keys()),p=list(self.probabilities_dict['decoding_order'].values()))
                    focus_sequence_cluster = "uniref100"
                    focus_sequence = row[col2id[self.columns['focus']]][:self.max_sequence_length_dict['focus']]
                    if len(focus_sequence)  == 0:
                        raise Exception(f'WARNING:focus_sequence:{focus_sequence} has length 0')
                        

                    if is_pair:
                        condition_sequences = []
                        prev_condition_clusters  = []
                        while not len(condition_sequences):
                            clusters = list(set(self.probabilities_dict['cluster'].keys()).difference(set(prev_condition_clusters)))
                            if len(clusters) == 0:
                                print(f'WARNING: Exhausted all cluster columns while smapling for a new cluster column. Treating sample as singleton')
                                condition_sequence = ""
                                break
                            probs = [1/len(clusters) for c in clusters]
                            condition_sequence_cluster = np.random.choice(clusters,p=probs)
                            print(f'INFO:Sampled Condition Cluster: {condition_sequence_cluster}')
                            print("self.columns['condition']:",self.columns['condition'])
                            condition_sequences = row[col2id[self.columns['condition'][condition_sequence_cluster]]].split(self.separator_token)
                            
                            if len(condition_sequences)  == 1:
                                condition_sequence  = condition_sequences[0]
                            elif len(condition_sequences) > 1:
                                condition_sequence_index = np.random.choice([i for i in range(len(condition_sequences))])
                                condition_sequence = condition_sequences[condition_sequence_index]
                            else:
                                print(f'WARNING:sampling for {condition_sequence_cluster} found no condition_sequences. choosing another column')   
                        sample = {"index": index,"focus":focus_sequence,"condition": condition_sequence,"focus_cluster":focus_sequence_cluster,"condition_cluster":condition_sequence_cluster,"decoding_order": decoding_order }
                    else:
                        #single sequence i.e only focus sequence and no condition sequence
                        sample = {"index": index,"focus":focus_sequence,"condition": "","focus_cluster":focus_sequence_cluster,"condition_cluster":"","decoding_order": decoding_order }
                    if len(sample['focus']) and not len(sample['focus_cluster']):
                        print(f'WARNING:FOCUS_CLUSTER->Invalid sample has focus_sequence:{sample['focus']} but no focus_cluster in sample:{sample['focus_cluster']}')
                    
                    if len(sample['condition']) and not len(sample['condition_cluster']):
                        print(f'WARNING:CONDITION_CLUSTER->Invalid sample has cond_sequence:{sample['condition']} but no condition_cluster in sample:{sample['condition_cluster']}')
                    
                    print(f'INFO: idx:{idx} sample:{sample}')
                    if self.tokenization_strategy == "otf":
                        masked_seqs_tokenized,seqs_tokenized,sentinel_indices,attention_mask = self.tokenizer(
                        sample['condition'],
                        sample['focus'],
                        sample['condition_cluster'],
                        sample['focus_cluster'],
                        sample['decoding_order'],
                        max_cond_sequence_length=self.max_sequence_length_dict['condition'],
                        max_focus_sequence_length=self.max_sequence_length_dict['focus'],
                        max_sequence_length=self.max_sequence_length_dict['total'],
                        add_special_tokens=True,
                        multisequence=True,
                        return_tensor=True,)
                        print(f'seqs_tokenized:{seqs_tokenized}')
                        print(f'masked_seqs_tokenized:{masked_seqs_tokenized}')
                        sample = {'input_ids': masked_seqs_tokenized, 'sentinel_indices': sentinel_indices, 'attention_mask': attention_mask,"labels": seqs_tokenized}
                        print('otf:',sample)
                        
                        yield sample
                    else:
                        yield sample
            
