from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import argparse

def parse_args ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='EdoAbati/whisper-medium-it-2')
    parser.add_argument('--dataset_name_or_path', type=str, default='RiTA-nlp/italic-easy')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--max_input_length_in_seconds', type=float, default=15)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    return parser.parse_args()

args = parse_args()

# -------------------------------- Loading dataset --------------------------------
italic = load_dataset(args.dataset_name_or_path, use_auth_token=True)
all_columns = italic["train"].column_names
col_to_remove = [ c for c in all_columns if c not in ["audio", "utt"] ]
italic = italic.remove_columns(col_to_remove)

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language="Italian", task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language="Italian", task="transcribe")
italic = italic.cast_column("audio", Audio(sampling_rate=16000))

# -------------------------------- Preprocessing --------------------------------
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()

def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["utt"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

italic = italic.map(prepare_dataset, remove_columns=italic.column_names["train"], num_proc=16)


def is_audio_in_length_range(length):
    return length < args.max_input_length_in_seconds

print ("Length of dataset before filtering: ", len(italic["train"]), " samples")
italic["train"] = italic["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)
print("Length of dataset after filtering: ", len(italic["train"]), " samples")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# evaluate with the 'normalised' WER
do_normalize_eval = True

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

safe_model_name = args.model_name_or_path.replace("/", "-")

if "easy" in args.dataset_name_or_path:
    output_dir = f"models/easy/{safe_model_name}"
elif "speaker" in args.dataset_name_or_path:
    output_dir = f"models/speaker/{safe_model_name}"
elif "noisy" in args.dataset_name_or_path:
    output_dir = f"models/noisy/{safe_model_name}"
else:
    output_dir = f"models/{safe_model_name}"


total_steps = ((len(italic["train"]) // args.batch_size) // args.gradient_accumulation_steps) * args.num_train_epochs

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=max(1, args.batch_size // 4),
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    warmup_steps=500,
    num_train_epochs=args.num_train_epochs,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=italic["train"],
    eval_dataset=italic["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()

model_name_repo = ""

if "medium" in args.model_name_or_path:
    model_name_repo = "Whisper Medium"
elif "small" in args.model_name_or_path:
    model_name_repo = "Whisper Small"
elif "large" in args.model_name_or_path:
    model_name_repo = "Whisper Large"
else:
    model_name_repo = "Whisper"


if "easy" in args.dataset_name_or_path:
    model_name_repo += " - ITALIC Easy"
    dataset_name_repo = "Italic Easy"
elif "noisy" in args.dataset_name_or_path:
    model_name_repo += " - ITALIC Noisy"
    dataset_name_repo = "Italic Noisy"
elif "speaker" in args.dataset_name_or_path:
    model_name_repo += " - ITALIC Speaker"
    dataset_name_repo = "Italic Speaker"
else:
    model_name_repo += " - ITALIC"

kwargs = {
    "dataset_tags": args.dataset_name_or_path,
    "dataset": dataset_name_repo,
    "language": "it",
    "model_name": model_name_repo,
    "finetuned_from": args.model_name_or_path,
    "tasks": "automatic-speech-recognition",
    "tags": "whisper,it,asr",
}

# get the best model from the training
model = trainer.model
model.save_pretrained(training_args.output_dir + "/best_model", **kwargs)
processor.save_pretrained(training_args.output_dir + "/best_model/", **kwargs)