from typing import Callable
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split

class ClusterDataset(Dataset):
    def __init__(
            self, 
            dataset_path: str, 
            cluster_table_path: str,
            subsample_size:int,
            val_size:int,
            test_size:int,
            size_to_sample_prob: Callable = lambda x: x,
            seed: int = 42,    
        ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.cluster_table_path = cluster_table_path
        self.cluster_to_seqs = {}
        self.cluster_table = pd.read_csv(
            cluster_table_path, dtype={'cluster_name': str, 'cluster_size': int}
        ).sample(n=subsample_size)
        self.cluster_table,self.cluster_table_eval = train_test_split(self.cluster_table,test_size = val_size + test_size)
        self.cluster_table_eval,self.cluster_table_test = train_test_split(self.cluster_table_eval,test_size = test_size)
        
        self.cluster_table['sample_prob'] = self.cluster_table['cluster_size'].apply(size_to_sample_prob)
        self.cluster_table['sample_prob'] /= self.cluster_table['sample_prob'].sum()
        
        self.cluster_table_test['sample_prob'] = self.cluster_table_test['cluster_size'].apply(size_to_sample_prob)
        self.cluster_table_test['sample_prob'] /= self.cluster_table_test['sample_prob'].sum()

        self.cluster_table_eval['sample_prob'] = self.cluster_table_eval['cluster_size'].apply(size_to_sample_prob)
        self.cluster_table_eval['sample_prob'] /= self.cluster_table_eval['sample_prob'].sum()

        self.test_size = test_size
        self.val_size = val_size
        self.generator = np.random.default_rng(seed)
        print(self.cluster_table.shape,self.cluster_table_eval.shape,self.cluster_table_test.shape)

    def __len__(self) -> int:
        return len(self.cluster_table)
    
    def get_cluster_seqs(self, cluster_path: str) -> list:
        if cluster_path not in self.cluster_to_seqs:
            self.cluster_to_seqs[cluster_path] = [
                str(x.seq) for x in SeqIO.parse(cluster_path, 'fasta')
            ]
        return self.cluster_to_seqs[cluster_path]

    def __iter__(self):
        for _ in range(len(self)):
            cluster_name = self.cluster_table.sample(
                n=1, weights='sample_prob', random_state=self.generator
            )[['cluster_name']].values[0][0]
            # Now we map cluster_name to the folder it is in
            if cluster_name == "unk":
                cluster_path = os.path.join(self.dataset_path, "unk", "unk.fasta")
            else:
                cluster_dir = f"{int(cluster_name) // 1000}000"
                cluster_path = os.path.join(self.dataset_path, cluster_dir, f"{cluster_name}.fasta")
            seqs = self.get_cluster_seqs(cluster_path)
            yield {"sequence":seqs[self.generator.integers(len(seqs))]}
    def test__iter__(self):
        for _ in range(self.test_size):
            cluster_name = self.cluster_table_test.sample(
                n=1, weights='sample_prob', random_state=self.generator
            )[['cluster_name']].values[0][0]
            # Now we map cluster_name to the folder it is in
            if cluster_name == "unk":
                cluster_path = os.path.join(self.dataset_path, "unk", "unk.fasta")
            else:
                cluster_dir = f"{int(cluster_name) // 1000}000"
                cluster_path = os.path.join(self.dataset_path, cluster_dir, f"{cluster_name}.fasta")
            seqs = self.get_cluster_seqs(cluster_path)
            yield {"sequence":seqs[self.generator.integers(len(seqs))]}
    def val__iter__(self):
        for _ in range(self.val_size):
            cluster_name = self.cluster_table_eval.sample(
                n=1, weights='sample_prob', random_state=self.generator
            )[['cluster_name']].values[0][0]
            # Now we map cluster_name to the folder it is in
            if cluster_name == "unk":
                cluster_path = os.path.join(self.dataset_path, "unk", "unk.fasta")
            else:
                cluster_dir = f"{int(cluster_name) // 1000}000"
                cluster_path = os.path.join(self.dataset_path, cluster_dir, f"{cluster_name}.fasta")
            seqs = self.get_cluster_seqs(cluster_path)
            yield {"sequence":seqs[self.generator.integers(len(seqs))]}
