import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from datasets import load_dataset
import pandas as pd
import argparse

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
        default=8, 
        type=int,
        required=False)
    parser.add_argument(
        "--epochs",
        help="number of training epochs",
        default=30, 
        type=int,
        required=False)
    parser.add_argument(
        "--model",
        help="model to use -- choose one of: facebook/wav2vec2-large-xlsr-53, \
            facebook/wav2vec2-xls-r-300m, facebook/wav2vec2-xls-r-1b, \
            facebook/wav2vec2-xls-r-2b, jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        default="facebook/wav2vec2-xls-r-300m",  
        type=str,                          
        required=True)                     
    parser.add_argument(
        "--dataset_name",
        help="name of the dataset to use",
        default="rita-nlp/italic-easy",
        type=str,
        required=True)
    parser.add_argument(
        "--use_auth_token",
        help="use authentication token for dataset download",
        action='store_true',
        required=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int,
    )
    args = parser.parse_args()
    return args


""" Define model and feature extractor """
def define_model(model_checkpoint, num_labels, label2id, id2label):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )
    return feature_extractor, model


""" Main Program """
if __name__ == '__main__':

    ## Multiprocessing 
    torch.multiprocessing.set_start_method('spawn')

    ## Utils 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    max_duration = 10.0 
    dataset_name = parse_cmd_line_params().dataset_name
    use_auth_token = parse_cmd_line_params().use_auth_token
    batch_size = parse_cmd_line_params().batch
    num_epochs = parse_cmd_line_params().epochs
    model_checkpoint = parse_cmd_line_params().model
    gradient_accumulation_steps = parse_cmd_line_params().gradient_accumulation_steps
    model_name = model_checkpoint.split("/")[-1]
    save_dataset_name = dataset_name.split("italic-")[-1]
    output_dir = model_name + "-ic-finetuning-" + save_dataset_name

    ## Load Dataset
    dataset = load_dataset(
        dataset_name, 
        use_auth_token=True if use_auth_token else None
    )
    ds_train = dataset["train"]
    ds_validation = dataset["validation"]

    ## Mapping intents to labels
    intents = set(ds_train['intent'])
    label2id, id2label = dict(), dict()
    for i, label in enumerate(intents):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Model & Feature Extractor
    model_checkpoint = parse_cmd_line_params().model
    model_name = model_checkpoint.split("/")[-1]
    feature_extractor, model = define_model(model_checkpoint, num_labels, label2id, id2label)

    ## Train & Validation Datasets 
    train_dataset = Dataset(ds_train, feature_extractor, label2id, max_duration, device)
    val_dataset = Dataset(ds_validation, feature_extractor, label2id, max_duration, device)

    ## Training Arguments and Class Weights
    training_arguments = define_training_args(output_dir, batch_size, num_epochs, gradient_accumulation_steps=gradient_accumulation_steps)
    # class_weights = compute_class_weights(ds_train, label2id)

    ## Trainer 
    trainer = WeightedTrainer(
        # class_weights=class_weights,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    ## Train and Evaluate
    trainer.train()
    trainer.save_model(output_dir)
    trainer.evaluate()