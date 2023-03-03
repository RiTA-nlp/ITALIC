import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from datasets import load_dataset
import pandas as pd
import argparse
import os
import json

import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset
from utils import WeightedTrainer, define_training_args, \
    compute_metrics
    

""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="batch_size")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=32, 
        type=int,
        required=False)
    parser.add_argument(
        "--model_checkpoint",
        help="model ckpt to evaluate",
        default="wav2vec2-xls-r-300m-ic-finetuning-easy",  
        type=str,                          
        required=True)   
    parser.add_argument(
        "--feature_extractor",
        help="feature extractor to use",
        default="facebook/wav2vec2-xls-r-300m",
        type=str,
        required=True)                  
    parser.add_argument(
        "--dataset_name",
        help="name of the dataset to use",
        default="RiTA-nlp/italic-easy",
        type=str,
        required=True)
    parser.add_argument(
        "--use_auth_token",
        help="use authentication token for dataset download",
        action='store_true',
        required=False)
    args = parser.parse_args()
    return args


""" Define model and feature extractor """
def define_model(model_checkpoint, feature_extractor):
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        local_files_only=True
    )
    return feature_extractor, model


""" Main Program """
if __name__ == '__main__':

    ## Multiprocessing 
    torch.multiprocessing.set_start_method('spawn')

    ## Utils 
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    max_duration = 10.0 
    dataset_name = parse_cmd_line_params().dataset_name
    use_auth_token = parse_cmd_line_params().use_auth_token
    batch_size = parse_cmd_line_params().batch
    model_checkpoint = parse_cmd_line_params().model_checkpoint
    feature_extractor = parse_cmd_line_params().feature_extractor
    model_name = model_checkpoint.split("/")[-1]
    save_dataset_name = dataset_name.split("italic-")[-1]
    output_dir = "inference_results/" + model_name + "_" + save_dataset_name + "/"

    ## Load Dataset
    dataset = load_dataset(
        dataset_name, 
        use_auth_token=True if use_auth_token else None
    )
    ds_test = dataset["test"]

    ## Model & Feature Extractor
    feature_extractor, model = define_model(
        model_checkpoint, 
        feature_extractor
        )
 
    ## Label2ID from model_ckpt
    with open(os.path.join(model_checkpoint, "config.json"), "r") as f:
        config = json.load(f)
        label2id = config["label2id"]

    ## Test Dataset
    test_dataset = Dataset(
        ds_test, 
        feature_extractor, 
        label2id, 
        max_duration, 
        device
        )

    ## Test Arguments
    test_arguments = define_training_args(
        output_dir, 
        batch_size 
        )

    ## Trainer
    trainer = WeightedTrainer(
        model=model,
        args=test_arguments,
        compute_metrics=compute_metrics
    )

    ## Inference
    test_results = trainer.predict(test_dataset)
    print(test_results.metrics)