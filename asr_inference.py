import argparse

from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset, Audio
import evaluate
from tqdm import tqdm

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    return sample["utt"]

whisper_norm = BasicTextNormalizer()


def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}


def main(args):

    print (f"Evaluating {args.model_id} on {args.dataset} ({args.split})...")


    batch_size = args.batch_size
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=args.model_id, device=args.device
    )

    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )
    )

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        use_auth_token=True,
    )

    # Only uncomment for debugging
    # dataset = dataset.take(args.max_eval_samples)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    predictions = []
    references = []

    # run streamed inference
    for out in tqdm(whisper_asr(data(dataset), batch_size=batch_size)):
        predictions.append(whisper_norm(out["text"]))
        references.append(out["reference"][0])

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)

    print("WER:", wer)
    print("CER:", cer)
    print("Done!\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="RiTA-nlp/italic-easy",
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'`",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Two letter language code for the transcription language, e.g. use 'en' for English.",
    )
    args = parser.parse_args()

    main(args)