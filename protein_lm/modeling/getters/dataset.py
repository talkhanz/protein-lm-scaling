from typing import Dict, Literal, Optional,Union

from datasets import Dataset, load_dataset,IterableDataset
from datasets.dataset_dict import DatasetDict,IterableDatasetDict
from pydantic import BaseModel

from protein_lm.dataset.cluster_dataset import ClusterDataset
from protein_lm.dataset.multisequence_dataset import MutliSequenceIterableDataset,MutliSequenceIterableDataset2


import torch
class DatasetConfig(BaseModel):
    dataset_type: Literal["csv", "huggingface","colabfold","paired","multisequence"]

    # The path if local or the huggingface dataset name if huggingface
    dataset_loc: str
    test_dataset_loc: str
    val_dataset_loc: str

    #This is cluster_table for ClusterDataset when dataset_type is colabfold
    cluster_loc: Optional[str] = None

    # sample size to limit to, if any, usually for debugging
    subsample_size: Optional[Union[int,float]] = None

    """
    Args for splitting into train, val, test
    to be updated once we have more options
    """
    # split seed
    split_seed: Optional[int] = None
    # size of validation dataset
    val_size: Optional[Union[int,float]] = None
    # size of test dataset
    test_size: Optional[Union[int,float]] = None

    # name of the column that contains the sequence
    sequence_column_name: str
    index_column_name: Optional[str] = None
    cond_sequence_column_names: Optional[list] = None
    focus_sequence_column_name: Optional[str] = None
    pair_prob: Optional[str] = None
   

    max_cond_sequence_length: Optional[int] = None
    max_focus_sequence_length: Optional[int] = None
    max_sequence_length: int = None
    cond_sequence_mask_fraction: Optional[Union[int,float]] = 0.25
    focus_sequence_mask_fraction:Optional[Union[int,float]]= 0.0
    tokenization_strategy: Optional[str] = "precompute"


def set_input_ids(
    dataset_type = "",
    result=None,
    tokenizer=None,
    sequence_column_name="sequence",
    cond_sequence_column_name="condition",
    focus_sequence_column_name="focus",
    decoding_order_column_name = "decoding_order",
    max_cond_sequence_length=1024,
    max_focus_sequence_length=1024,
    max_sequence_length=1024,
):
    
    if dataset_type == "multisequence":
        seqs_tokenized,sentinel_indices = tokenizer(
            result[cond_sequence_column_name],
            result[focus_sequence_column_name],
            result[cond_sequence_column_name + "_cluster"],
            result[focus_sequence_column_name + "_cluster"],
            result[decoding_order_column_name],
            max_cond_sequence_length=max_cond_sequence_length,
            max_focus_sequence_length=max_focus_sequence_length,
            max_sequence_length=max_sequence_length,
            add_special_tokens=True,
            multisequence=True,
            return_tensors=True,)
        result['input_ids'] = seqs_tokenized
        result['sentinel_indices'] = sentinel_indices
    
    else:
        result["input_ids"] = tokenizer(
            result[sequence_column_name],
            max_sequence_length=max_sequence_length,
            add_special_tokens=True,
            return_tensors=True,
        )
    print('###result#####')
    print(result)
    return result

def batch_set_curriculum_learning_column(
    result=None,
    input_column_name='sequence',
    curriculum_learning_column_name='sequence_length',
    strategy='sequence_length'
):
    supported_strategies = ['sequence_length', 'ppl', 'plddt']

    if strategy not in supported_strategies:
        raise Exception(f'Invalid {strategy} provided. Supported strategy values include {", ".join(supported_strategies)}')

    if strategy == 'sequence_length':
        # LengthGroupedSampler sorts in descending so we make it ascending by multiplying with -1
        result[curriculum_learning_column_name] = [-len(x) for x in result[input_column_name]]
    elif strategy in ['ppl', 'plddt']:
        result[curriculum_learning_column_name] = [-x for x in result[strategy]]

    return result

def set_labels(result):
    result["labels"] = result["input_ids"].copy()
    return result


def train_val_test_split(
    dataset_dict: DatasetDict,
    config: DatasetConfig,
) -> DatasetDict:
    """
    Given a dictionary of datasets that only contains the split "train",
    optionally subsamples it, and then splits it
    so that it has potentially 3 splits: "train", "val", "test", where
    "val" and "test" splits do not exist if the specified sizes are 0
    """
    assert set(dataset_dict.keys()) == {
        "train"
    }, f"{train_val_test_split.__name__} expects its input to have the keys \
        ['train'] but the input has keys {list(dataset_dict.keys())}"

    dataset = dataset_dict["train"]
    val_size = config.val_size
    test_size = config.test_size

    assert isinstance(
        dataset, Dataset
    ), f"Invalid dataset type {type(dataset)}, only datasets.Dataset allowed"

    dataset = dataset.shuffle(seed=config.split_seed)
    dataset_size = dataset.shape[0]
    if config.subsample_size is not None:
        if isinstance(config.subsample_size,float):
            assert config.subsample_size >= 0 and config.subsample_size <= 1
            subsample_size = int(config.subsample_size * dataset_size)
        dataset = dataset.select(range(subsample_size))

    if isinstance(val_size,float):
        assert val_size >= 0 and val_size <= 1
        val_size = int(val_size * dataset_size)

    if isinstance(test_size,float):
        assert test_size >= 0 and test_size <= 1
        test_size = int(test_size * dataset_size)
    valtest_size = val_size + test_size

    if valtest_size > 0:
        train_valtest = dataset.train_test_split(
            test_size=val_size + test_size,
            shuffle=False,
        )
        split_dict = {
            "train": train_valtest["train"],
        }
        if test_size > 0 and val_size > 0:
            test_val = train_valtest["test"].train_test_split(
                test_size=test_size,
                shuffle=False,
            )
            split_dict["val"] = test_val["train"]
            split_dict["test"] = test_val["test"]
        elif val_size > 0:
            split_dict["val"] = train_valtest["test"]
        else:
            split_dict["test"] = train_valtest["test"]
    else:
        split_dict = {
            "train": dataset,
        }

    split_dataset_dict = DatasetDict(split_dict)
    return split_dataset_dict


def get_csv_dataset(config: DatasetConfig) -> Dataset:
    # note that a csv is read as having just one split "train"
    dataset_dict = load_dataset("csv", data_files=config.dataset_loc)
    return train_val_test_split(dataset_dict, config)


def get_huggingface_dataset(config: DatasetConfig) -> Dataset:
    dataset_dict = load_dataset(config.dataset_loc)
    if set(dataset_dict.keys()) == {"train", "val", "test"}:
        return dataset_dict

    assert set(dataset_dict.keys()) == {
        "train"
    }, f"Huggingface DatasetDicts should have the keys {{'train'}} or \
        {{'train', 'val', 'split'}} but this DatasetDict has keys \
            {set(dataset_dict.keys())}"
    return train_val_test_split(dataset_dict, config)

def get_colabfold_dataset(config:DatasetConfig) -> Dataset:
    ds = ClusterDataset(dataset_path = config.dataset_loc, cluster_table_path = config.cluster_loc,subsample_size=config.subsample_size,val_size = config.val_size,test_size = config.test_size)
    ds = DatasetDict({"train":  Dataset.from_generator(ds.__iter__,gen_kwargs={"split": "train"}),"test": Dataset.from_generator(ds.__iter__,gen_kwargs={"split": "test"}),"val":Dataset.from_generator(ds.__iter__,gen_kwargs={"split": "val"})})
    return ds

def get_multisequence_dataset(config:DatasetConfig,tokenizer) -> Dataset:
    exception_msg = f"sum of maximum condition sequence length {config.max_cond_sequence_length} and maximum focus sequence length {config.max_focus_sequence_length} is greater than maximum_sequence_length:{config.max_sequence_length}"
    assert config.max_cond_sequence_length + config.max_focus_sequence_length <= config.max_sequence_length,exception_msg
    torch_ds = MutliSequenceIterableDataset(
            dataset_path=config.dataset_loc, 
            cluster_table_path = "",
            index_column = config.index_column_name,
            focus_sequence_column = config.focus_sequence_column_name,
            condition_sequence_columns = {"uniref30":config.cond_sequence_column_names[0],"uniref50":config.cond_sequence_column_names[1],"uniref90":config.cond_sequence_column_names[2],"struct":config.cond_sequence_column_names[3]}, 
            cond_sequence_length= config.max_cond_sequence_length,
            focus_sequence_length = config.max_focus_sequence_length,
            seed=config.split_seed,
            pair_prob= 0.7, #probably a sample will have a pair of sequences
            cluster_prob ={'uniref30': 0.25,'uniref50': 0.25,'uniref90':0.25,'struct': 0.25}, #if a pair is selected, probably of condition sequence per cluster
            decoding_order_prob = {'lr': 0.25,'rl': 0.25, 'fim': 0.25,'mo': 0.25}, #independant of paired sequence or singleton
            separator_token=",", #the token which separates the list of sequences in the condition sequence columns,
            tokenization_strategy = config.tokenization_strategy,
            tokenizer = tokenizer,
            )

    torch_test_ds = MutliSequenceIterableDataset(
            dataset_path=config.test_dataset_loc, 
            cluster_table_path = "",
            index_column = config.index_column_name,
            focus_sequence_column = config.focus_sequence_column_name,
            condition_sequence_columns = {"uniref30":config.cond_sequence_column_names[0],"uniref50":config.cond_sequence_column_names[1],"uniref90":config.cond_sequence_column_names[2],"struct":config.cond_sequence_column_names[3]}, 
            cond_sequence_length= config.max_cond_sequence_length,
            focus_sequence_length = config.max_focus_sequence_length,
            seed=config.split_seed,
            pair_prob= 0.7, #probably a sample will have a pair of sequences
            cluster_prob ={'uniref30': 0.25,'uniref50': 0.25,'uniref90':0.25,'struct': 0.25}, #if a pair is selected, probably of condition sequence per cluster
            decoding_order_prob = {'lr': 0.25,'rl': 0.25, 'fim': 0.25,'mo': 0.25}, #independant of paired sequence or singleton
            separator_token=",", #the token which separates the list of sequences in the condition sequence columns
            tokenization_strategy = config.tokenization_strategy,
            tokenizer = tokenizer,
            )
    
    torch_val_ds = MutliSequenceIterableDataset(
            dataset_path=config.val_dataset_loc, 
            cluster_table_path = "",
            index_column = config.index_column_name,
            focus_sequence_column = config.focus_sequence_column_name,
            condition_sequence_columns = {"uniref30":config.cond_sequence_column_names[0],"uniref50":config.cond_sequence_column_names[1],"uniref90":config.cond_sequence_column_names[2],"struct":config.cond_sequence_column_names[3]}, 
            cond_sequence_length= config.max_cond_sequence_length,
            focus_sequence_length = config.max_focus_sequence_length,
            seed=config.split_seed,
            pair_prob= 0.7, #probably a sample will have a pair of sequences
            cluster_prob ={'uniref30': 0.25,'uniref50': 0.25,'uniref90':0.25,'struct': 0.25}, #if a pair is selected, probably of condition sequence per cluster
            decoding_order_prob = {'lr': 0.25,'rl': 0.25, 'fim': 0.25,'mo': 0.25}, #independant of paired sequence or singleton
            separator_token=",", #the token which separates the list of sequences in the condition sequence columns
            tokenization_strategy = config.tokenization_strategy,
            tokenizer = tokenizer,
            )
            

    def torch_ds_generator(split = "train"):
        if split == "train":
            for item in torch_ds:
                yield item  # this has to be a dictionary
        elif split == "test":
            for item in torch_test_ds:
                yield item  # this has to be a dictionary
        elif split == "val":
            for item in torch_val_ds:
                yield item  # this has to be a dictionary

    
        
   
    train_ds = IterableDataset.from_generator(torch_ds_generator,gen_kwargs={"split": "train"})
    test_ds = IterableDataset.from_generator(torch_ds_generator,gen_kwargs={"split": "test"})
    val_ds = IterableDataset.from_generator(torch_ds_generator,gen_kwargs={"split": "val"})
    
    # ds = ds.with_format("torch") # converts back to torch dataset
    dataset_dict = IterableDatasetDict({"train": train_ds,"test": test_ds,"val": val_ds})
    print(f'multisequence:{dataset_dict}') 
    return dataset_dict

def get_dataset(config_dict: Dict, tokenizer) -> Dataset:
    config = DatasetConfig(**config_dict)

    if config.dataset_type == "csv":
        train_ds = get_csv_dataset(config)
    elif config.dataset_type == "huggingface":
        train_ds = get_huggingface_dataset(config)
    elif config.dataset_type == "colabfold":
        train_ds = get_colabfold_dataset(config)
    elif config.dataset_type == "multisequence":
        train_ds = get_multisequence_dataset(config,tokenizer)
        
    else:
        raise ValueError(f"Invalid dataset_type {config.dataset_type}!")
    if config.tokenization_strategy == "precompute":    
        train_ds = train_ds.map(
                lambda e: set_input_ids(
                    dataset_type = config.dataset_type,
                    result=e,
                    tokenizer=tokenizer,
                    sequence_column_name=config.sequence_column_name,
                    max_cond_sequence_length=config.max_cond_sequence_length,
                    max_focus_sequence_length=config.max_focus_sequence_length,
                    max_sequence_length=config.max_sequence_length,
                ),
                batched=True)
        train_ds = train_ds.map(set_labels, batched=True)
        print('###get_dataset####')
        print(train_ds['train'].info)
    return train_ds
