import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import pandas as pd
import argparse

import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset
from utils import WeightedTrainer, define_training_args, \
    compute_metrics, compute_class_weights
    

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
        "--steps",
        help="number of training steps",
        default=2800, 
        type=int,
        required=False)
    parser.add_argument(
        "--model",
        help="model to use -- choose one of: facebook/wav2vec2-base, \
            facebook/wav2vec2-large-960h, facebook/wav2vec2-large-xlsr-53, \
            facebook/wav2vec2-xls-r-300m, facebook/wav2vec2-xls-r-1b, \
            facebook/wav2vec2-xls-r-2b, facebook/hubert-base-ls960, \
            facebook/hubert-large-ls960-ft, facebook/hubert-xlarge-ls960-ft" ,
        default="facebook/wav2vec2-base",  
        type=str,                          
        required=True)                     
    parser.add_argument(
        "--df_train",
        help="path to the train df",
        default="data/train_data.csv",
        type=str,
        required=True) 
    parser.add_argument(
        "--df_test",
        help="path to the test df",
        default="data/test_data.csv",
        type=str,
        required=True) 
    args = parser.parse_args()
    return args


""" Read and Process Data"""
def read_data(df_train_path, df_test_path):
    df_train = pd.read_csv(df_train_path, index_col=None)
    df_test = pd.read_csv(df_test_path, index_col=None)

    ## Prepare Labels
    intents = df_train['intent'].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(intents):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Train
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index,'intent']]
    df_train['label'] = df_train['label'].astype(int)
    df_train.to_csv('data/df_train.csv', index=False)

    ## Test
    for index in range(0,len(df_test)):
        df_test.loc[index,'label'] = label2id[df_test.loc[index,'intent']]
    df_test['label'] = df_test['label'].astype(int)
    df_test.to_csv('data/df_test.csv', index=False)

    return df_train, df_test, num_labels, label2id, id2label


""" Define model and feature extractor """
def define_model(model_checkpoint, num_labels, label2id, id2label):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        local_files_only=True
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
    batch_size = parse_cmd_line_params().batch
    num_steps = parse_cmd_line_params().steps
    model_checkpoint = parse_cmd_line_params().model
    model_name = model_checkpoint.split("/")[-1]
    output_dir = model_name + "-ic-finetuning"

    ## Train & Test df
    df_train, df_test, num_labels, label2id, id2label = read_data(
        parse_cmd_line_params().df_train,
        parse_cmd_line_params().df_test)

    ## Model & Feature Extractor
    model_checkpoint = parse_cmd_line_params().model
    model_name = model_checkpoint.split("/")[-1]
    feature_extractor, model = define_model(model_checkpoint, num_labels, label2id, id2label)

    ## Train & Test Datasets 
    train_dataset = Dataset(df_train, feature_extractor, max_duration, device)
    test_dataset = Dataset(df_test, feature_extractor, max_duration, device)

    ## Training Arguments and Class Weights
    training_arguments = define_training_args(output_dir, batch_size, num_steps)
    class_weights = compute_class_weights(df_train)

    ## Trainer 
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    ## Train and Evaluate
    trainer.train()
    trainer.save_model(output_dir)
    trainer.evaluate()