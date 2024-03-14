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
        self.cluster_table_dict = {"complete": None,"train": None , "test": None, "val": None}
        self.cluster_table_dict["complete"]  = pd.read_csv(
            cluster_table_path, dtype={'cluster_name': str, 'cluster_size': int}
        ).sample(n=subsample_size)
        self.cluster_table_dict["train"],self.cluster_table_dict["test"] = train_test_split(self.cluster_table_dict["complete"],test_size = test_size +val_size)
        self.cluster_table_dict["test"],self.cluster_table_dict["val"] = train_test_split(self.cluster_table_dict["test"],test_size = val_size)
        for split in ["complete","train","test","val"]:
            self.cluster_table_dict[split]['sample_prob'] = self.cluster_table_dict[split]['cluster_size'].apply(size_to_sample_prob)
            self.cluster_table_dict[split]['sample_prob'] /= self.cluster_table_dict[split]['sample_prob'].sum()
        self.generator = np.random.default_rng(seed)
        

    def __len__(self) -> int:
        return len(self.cluster_table_dict["complete"])
    
    def get_cluster_seqs(self, cluster_path: str) -> list:
        if cluster_path not in self.cluster_to_seqs:
            self.cluster_to_seqs[cluster_path] = [
                str(x.seq) for x in SeqIO.parse(cluster_path, 'fasta')
            ]
        return self.cluster_to_seqs[cluster_path]

    def __iter__(self,split = "train"):
        for _ in range(len(self.cluster_table_dict[split])):
            cluster_name = self.cluster_table_dict[split].sample(
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

