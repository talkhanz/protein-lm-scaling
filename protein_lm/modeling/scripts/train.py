import argparse
import math


import yaml
from transformers import Trainer

from protein_lm.modeling.getters.data_collator import get_data_collator
from protein_lm.modeling.getters.dataset import get_dataset
from protein_lm.modeling.getters.model import get_model
from protein_lm.modeling.getters.tokenizer import get_tokenizer
from protein_lm.modeling.getters.training_args import get_training_args
from protein_lm.modeling.getters.wandb_log import setup_wandb


def train(
    args: dict,
):
    """
    Main script to train APT.
    """
    print(args)
    with open(args['config_file'], "r") as cf:
        config_dict = yaml.safe_load(cf)
        print(config_dict)

    tokenizer = get_tokenizer(config_dict=config_dict["tokenizer"])
    dataset = get_dataset(
        config_dict=config_dict["dataset"],
        tokenizer=tokenizer,
    )

    model = get_model(
        config_dict=config_dict["model"],
    )
    model.train()

    data_collator = get_data_collator(
        config_dict=config_dict["data_collator"],
    )
    # if config_dict['dataset']['do_curriculum_learning']:
    #     #groupy_by_length uses the LengthGroupedSampler, 
    #     #we have precomputed the lengths (or any discrete column) which can be used as sampling criteria 
    #     config_dict["training_arguments"]['group_by_length'] = config_dict['dataset']['do_curriculum_learning']
    #     config_dict["training_arguments"]['length_column_name'] = config_dict['dataset']['curriculum_learning_column_name']
    
    training_args = get_training_args(
        config_dict=config_dict["training_arguments"],
    )

    if "wandb" in training_args.report_to and "wandb" in config_dict:
        wandb_dict = {
            "host": args["wandb_host"],
            "project": args["wandb_project"],
            "name": config_dict["wandb"]["name"],
            "dir": config_dict["wandb"]["dir"],
            "api_key":args["wandb_api_key"]
        }
        setup_wandb(wandb_dict)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("val", None),
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    args_dict = {
        "--config_file": {"default": "protein_lm/configs/train/toy_localcsv.yaml","type": str, "help": "Config yaml for training"},
        "--wandb_host": {"default": "", "type": str, "help":"Set this to the hostname you want to see in the wandb interface if you don't want to use the system provided hostname"},
        "--wandb_api_key": {"default": "", "type": str, "help":"Sets the authentication key associated with your account. You can find your key on your settings page. This must be set if wandb login hasn't been run on the remote machine."},
        "--wandb_project": {"default": "protein-lm-scaling", "type": str, "help":"The project associated with your run. This can also be set with wandb init, but the environmental variable will override the value."},
        "--wandb_name": {"default": "colabfold", "type": str, "help":"The human-readable name of your run. If not set it will be randomly generated by W&B"},
        "--wandb_dir": {"default": "protein_lm/experiments", "type": str, "help":"Set this to an absolute path to store all generated files here instead of the wandb directory relative to your training script. be sure this directory exists and the user your process runs as can write to it"} 
    }
    parser = argparse.ArgumentParser()
    for key,value in args_dict.items():
        parser.add_argument(
        key,
        default=value["default"],
        type=value["type"],
        help=value["help"],
    )

    args = parser.parse_args()
    train(args=vars(args))
