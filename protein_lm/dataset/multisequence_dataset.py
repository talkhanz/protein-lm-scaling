import torch
import pandas as pd
import os
import numpy as np
from typing import Union


class MutliSequenceDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path: str, 
            mapping_table_path: str,
            focus_sequence_column:str = "U100",
            condition_sequence_columns:dict = {'u50': 'UniRef50_clust_str','u90':'UniRef90_clust_str'}, 
            cond_sequence_length:int = 5,
            focus_sequence_length:int = 5,
            pair_prob:float = 0.7, #probably a sample will have a pair of sequences
            cluster_prob:dict = {'u50':0.5,'u90': 0.5}, #if a pair is selected, probably of condition sequence per cluster
            decoding_order_prob:dict = {'lr': 0.25,'rl': 0.25, 'fim': 0.25,'mo': 0.25}, #independant of paired sequence or singleton
            separator_token:str=",", #the token which separates the list of sequences in the condition sequence columns
            seed: int = 42,     
        ) -> None:
       
        

        self.dataset = pd.read_csv(dataset_path) # main dataset containing uniref 100 focus sequences and list of uniref 50 and 90 clustered sequences
        self.probabilities_dict = {"pair": pair_prob,'cluster': cluster_prob,'decoding_order':decoding_order_prob}
        self.columns = {"condition":condition_sequence_columns,"focus":focus_sequence_column}
        self.max_sequence_length_dict = {'condition': cond_sequence_length,'focus': focus_sequence_length}
        self.separator_token = separator_token
        if os.path.isfile(mapping_table_path):
            self.mapping_dataset = pd.read_csv(mapping_table_path) # a placeholder for any mapping table we may need in the future
        np.random.seed(seed)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.row = self.dataset.iloc[idx]
        is_pair = np.random.uniform() < self.probabilities_dict['pair']
        decoding_order = np.random.choice(list(self.probabilities_dict['decoding_order'].keys()),p=list(self.probabilities_dict['decoding_order'].values()))
        focus_sequence_cluster = "uniref100"
        focus_sequence = self.row[self.columns['focus']][:self.max_sequence_length_dict['focus']]
        if len(focus_sequence)  == 0:
            print(f'WARNING:focus_sequence:{focus_sequence} has length 0')

        if is_pair:   
            condition_sequence_cluster = np.random.choice(list(self.probabilities_dict['cluster'].keys()),p=list(self.probabilities_dict['cluster'].values()))
            condition_sequences = self.row[self.columns['condition'][condition_sequence_cluster]].split(self.separator_token)
            if len(condition_sequences)  == 1:
                condition_sequence  = condition_sequences[0]
            elif len(condition_sequences) > 1:
                condition_sequence_index = np.random.choice([i for i in range(len(condition_sequences))])
                condition_sequence = condition_sequences[condition_sequence_index]
            else:
                print(f'WARNING:Invalid sample in column:{condition_sequence_cluster}')
    
            sample = {"focus":focus_sequence,"condition": condition_sequence,"focus_cluster":focus_sequence_cluster,"condition_cluster":condition_sequence_cluster,"decoding_order": decoding_order }
        else:
            #single sequence i.e only focus sequence and no condition sequence
            sample = {"focus":focus_sequence,"condition": "","focus_cluster":focus_sequence_cluster,"condition_cluster":"","decoding_order": decoding_order }
        print(f'INFO: idx:{idx} sample:{sample}')
        return sample
        
