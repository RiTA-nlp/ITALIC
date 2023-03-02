import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import TrainingArguments, Trainer
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
        "--model_ckpt",
        help="ckpt model to evaluate",  
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



""" Main Program """
if __name__ == '__main__':

    ## Multiprocessing 
    # torch.multiprocessing.set_start_method('spawn')

    ## Utils 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    max_duration = 10.0 
    dataset_name = parse_cmd_line_params().dataset_name
    use_auth_token = parse_cmd_line_params().use_auth_token
    batch_size = parse_cmd_line_params().batch
    model_checkpoint = parse_cmd_line_params().model_ckpt
    feature_extractor = parse_cmd_line_params().feature_extractor
    output_dir = "inference_results"

    ## Model and Feature Extractor
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        local_files_only=True
        )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        feature_extractor
        )

    ## Load Dataset
    dataset = load_dataset(
        dataset_name, 
        use_auth_token=True if use_auth_token else None
    )
    ds_train = dataset["train"]
    ds_test = dataset["validation"]

    ## Mapping intents to labels
    intents = set(ds_train['intent'])
    label2id, id2label = dict(), dict()
    for i, label in enumerate(intents):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    test_dataset = Dataset(ds_test, 
        feature_extractor, 
        label2id, 
        max_duration, 
        device)

    ## Inference
    test_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        # warmup_ratio=0.1,
        # weight_decay=0.01,
        # logging_steps=50,
        # eval_steps=100,                 
        # save_steps=100,                 
        # save_total_limit=2,
        # load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=batch_size,   
        dataloader_drop_last=False    
        )
    # trainer = Trainer(
    #     model = model, 
    #     args = test_args, 
    #     compute_metrics = compute_metrics
    #     )
        ## Trainer 
    # test_args = define_training_args(
    #     output_dir, 
    #     batch_size, 
    #     num_epochs, 
    #     gradient_accumulation_steps=gradient_accumulation_steps)

    trainer = WeightedTrainer(
        model=model,
        args=test_args,
        compute_metrics=compute_metrics
    )

    test_results = trainer.predict(test_dataset)
    print(test_results.metrics)