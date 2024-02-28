import pandas as pd
import os 
from Bio import SeqIO
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
import multiprocessing
import sys

import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

task = sys.argv[1]
cpu_count = multiprocessing.cpu_count()
print(f'task:{task} cpu-count:{cpu_count}')
dataset_path = "/weka/home-othertea/protein-lm-scaling/colabfold_parsing/output"
cluster_path =  "/weka/home-othertea/protein-lm-scaling/colabfold_parsing/cluster_sizes.csv"
protein_gym_path = "/admin/home-talkhanz/repos/analysis/DMS_substitutions.csv"
protein_gym_updated_path = "/admin/home-talkhanz/repos/analysis/DMS_analysis.csv"
plot_path = "/admin/home-talkhanz/repos/analysis/plot.png"
seq_dict_path =  "/admin/home-talkhanz/repos/analysis/seq_dict_final.pkl"
count_path = "/admin/home-talkhanz/repos/analysis/count_dist.pkl"
pg_df = pd.read_csv(protein_gym_path)
target_seqs = pg_df.target_seq.unique().tolist()
save = True
id_list = []
unknown_id = 0
def default_value():
    return 0
def default_value_id():
    return {'cluster_size': 0 , 'cluster_id': ''}
target_seq_ht = dict(zip(target_seqs,['FOUND' for i in range(len(target_seqs))]))
count_found = defaultdict(default_value)

def get_filename(dataset_path,cluster,filename):
    if filename == 'unk':
        filepath = os.path.join(dataset_path,cluster,'unk.fasta')
    else:
        filepath = os.path.join(dataset_path,cluster,filename)
    return filepath

def parse_id(filepath):
    global unknown_id
    count_dist = defaultdict(default_value)
    cluster_id  = filepath.split('.fasta')[0].split('/')[-1]
    cluster_size = len([x for x in SeqIO.parse(filepath, 'fasta')])
    for x in SeqIO.parse(filepath, 'fasta'):
        x_id = str(x.id).lower()
        seq = str(x.seq)
        if 'uniref100' in x_id:
            count_dist['uniref100'] = count_dist['uniref100'] + 1
        elif 'bfd' in x_id:
            count_dist['bfd'] = count_dist['bfd'] + 1
        elif 'mgnify' in x_id:
            count_dist['mgnify'] = count_dist['mgnify'] + 1
        elif 'metaeuk' in x_id:
            count_dist['metaeuk'] = count_dist['metaeuk'] + 1
        elif 'smag' in x_id:
            count_dist['smag'] = count_dist['smag'] + 1
        elif 'topaz' in x_id:
            count_dist['topaz'] = count_dist['topaz'] + 1
        elif 'gpz' in x_id:
            count_dist['gpz'] = count_dist['gpz'] + 1
        elif 'metaclust' in x_id:
            count_dist['metaclust'] = count_dist['metaclust'] + 1  
        else:
            unknown_id = unknown_id + 1
            count_dist[x_id] = count_dist[x_id] + 1
            if unknown_id == 100 *1000:
                break
            
    
    return count_dist

def get_target_seq_cluster_info(filepath):
    global target_seq_ht
   
    cluster_id  = filepath.split('.fasta')[0].split('/')[-1]
    cluster_size = len([x for x in SeqIO.parse(filepath, 'fasta')])
    seq_dict = defaultdict(default_value_id)
    # print(f'########START#######')
    # print(f"get_target_seq_cluster_info({filepath})")
    
    for x in SeqIO.parse(filepath, 'fasta'):
        x_id = str(x.id).lower()
        seq = str(x.seq)
        try:
            val = target_seq_ht.get(seq,"NOT_FOUND")
            if val == "FOUND":
                seq_dict[seq]['cluster_id'] = cluster_id
                seq_dict[seq]['cluster_size'] = cluster_size
        except Exception as e:
            error = True
    # print(f'########END#######')
    return seq_dict

def save_variable(var = None , path = ''):
    print(f'saving variable to {path}')
    with open(path,'wb') as f:
        pickle.dump(var,f)

def load_variable(path = ''):
    print(f'loading variable from {path}')
    var = None
    with open(path,'rb') as f:
        var = pickle.load(f)
    return var

def save_plot(count_dist = None, plot_path = ''):
    plt.bar(list(count_dist.keys()),list(count_dist.values()))
    plt.xticks(rotation=90) 
    print(f'saving plot to {plot_path}')
    plt.savefig(plot_path, bbox_inches='tight')

paths = [get_filename(dataset_path,cluster,filename) for cluster in os.listdir(dataset_path) for filename in os.listdir(os.path.join(dataset_path,cluster))]
print('len(paths):',len(paths))
if task == 'target_seq':
    with tqdm_joblib(tqdm(desc=f"{task}", total=len(paths))) as progress_bar:
        seq_dicts = Parallel(n_jobs=cpu_count)(delayed(get_target_seq_cluster_info)(path) for path in paths)
    seq_final = {key:value for seq_dict in seq_dicts for key,value in seq_dict.items()}
    save_variable(var = seq_final , path = seq_dict_path)
elif task == "id_analysis":
    with tqdm_joblib(tqdm(desc=f"{task}", total=len(paths))) as progress_bar:
        count_dists = Parallel(n_jobs=cpu_count)(delayed(parse_id)(path) for path in paths)
    count_dist_final = defaultdict(default_value)
    for count_dist_dict in count_dists:
        for key,value in count_dist_dict.items():
            count_dist_final[key] = count_dist_final[key] + count_dist_dict[key]
    save_variable(var = count_dist_final , path = count_path)
    save_plot(count_dist = count_dist_final, plot_path = plot_path)


