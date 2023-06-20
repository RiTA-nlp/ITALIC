# ITALIC: An ITALian Intent Classification Dataset

This repository contains the code and the dataset for the paper [ITALIC: An ITALian Intent Classification Dataset](#).

ITALIC is a new intent classification dataset for the Italian language, which is the first of its kind. It includes spoken and written utterances and is annotated with 60 intents. The dataset is available on [Zenodo](https://zenodo.org/record/8040649) and we are working on making it accessible also from HuggingFace Hub.

### Latest Updates

- **June 15th, 2023**: ITALIC dataset has been released on [Zenodo](https://zenodo.org/record/8040649): https://zenodo.org/record/8040649.

## Table of Contents

- [Data collection](#data-collection)
- [Dataset](#dataset)
- [Usage](#usage)
- [Models used in the paper](#models-used-in-the-paper)
  - [SLU intent classification](#slu-intent-classification)
  - [ASR](#asr)
  - [NLU intent classification](#nlu-intent-classification)
- [Citation](#citation)
- [License](#license)

## Data collection

The data collection follows the MASSIVE NLU dataset which contains an annotated textual dataset for 60 intents. The data collection process is described in the paper [Massive Natural Language Understanding](https://arxiv.org/abs/2204.08582).

Following the MASSIVE NLU dataset, a pool of 70+ volunteers has been recruited to annotate the dataset. The volunteers were asked to record their voice while reading the utterances (the original text is available on MASSIVE dataset). Together with the audio, the volunteers were asked to provide a self-annotated description of the recording conditions (e.g., background noise, recording device). The audio recordings have also been validated and, in case of errors, re-recorded by the volunteers.

All the audio recordings included in the dataset have received a validation from at least two volunteers. All the audio recordings have been validated by native italian speakers (self-annotated).

## Dataset

The dataset is available on the [Zenodo](https://zenodo.org/record/8040649). It is composed of 3 different splits:
- easy: all the utterances are randomly shuffled and divided into 3 splits (train, validation, test).
- speaker: the utterances are divided into 3 splits (train, validation, test) based on the speaker. Each split only contains utterances from a pool of speakers that do not overlap with the other splits.
- noisy: the utterances are divided into 3 splits (train, validation, test) based on the recording conditions. The test split only contains utterances with the highest level of noise.

Each split contains the following annotations:
- `utt`: the original text of the utterance.
- `audio`: the audio recording of the utterance.
- `intent`: the intent of the utterance.
- `speaker`: the speaker of the utterance. The speaker is identified by a unique identifier and has been anonymized.
- `age`: the age of the speaker.
- `is_native`: whether the speaker is a native italian speaker or not.
- `gender`: the gender of the speaker (self-annotated).
- `region`: the region of the speaker (self-annotated).
- `nationality`: the nationality of the speaker (self-annotated).
- `lisp`: any kind of lisp of the speaker (self-annotated). It can be empty in case of no lisp.
- `education`: the education level of the speaker (self-annotated).
- `environment`: the environment of the recording (self-annotated). 
- `device`: the device used for the recording (self-annotated).


## Usage

The dataset can be loaded using the `datasets` library:

```python
from datasets import load_dataset
...
# complete information will be provided upon publication
```

The dataset has been designed for intent classification tasks. The `intent` column can be used as the label. However, the dataset can be used for other tasks as well. 

- **Intent classification**: the `intent` column can be used as the label.
- **Speaker identification**: the `speaker` column can be used as the label.
- **Automatic speech recognition**: the `utt` column can be used as the label.
- **Accent identification**: the `region` column can be used as the label.

For more information about the dataset, please refer to the [paper](#).


## Models used in the paper

### Hardware settings
All experiments were conducted on a private workstation with Intel Core i9-10980XE CPU, 1 $\times$ NVIDIA RTX A6000 GPU, 64 GB of RAM running Ubuntu 22.04 LTS.

### Parameter settings

The parameters used for the training of the models are set to allow a fair comparison between the different models and to follow the recommendations of the related literature. The parameters are summarized in the following table:

| Model | Task | Parameters | Learning rate | Batch size | Max epochs | Warmup | Weight decay | Avg. training time | Avg. inference time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| facebook/wav2vec2-xls-r-300m | SLU | 300M | 1e-4 | 128 | 30 | 0.1 ratio | 0.01 | 9m 35s per epoch | 13ms per sample |
| facebook/wav2vec2-xls-r-1b | SLU | 1B | 1e-4 | 32 | 30 | 0.1 ratio | 0.01 | 21m 30s per epoch | 29ms per sample |
| jonatasgrosman/wav2vec2-large-xlsr-53-italian | SLU | 300M | 1e-4 | 128 | 30 | 0.1 ratio | 0.01 | 9m 35s per epoch | 13ms per sample |
| jonatasgrosman/wav2vec2-xls-r-1b-italian | SLU | 1B | 1e-4 | 32 | 30 | 0.1 ratio | 0.01 | 21m 30s per epoch | 29ms per sample |
| ALM/whisper-it-small-augmented | ASR | 224M | 1e-5 | 8 | 5 | 500 steps | 0.01 | 26m 30s per epoch | 25ms per sample |
| EdoAbati/whisper-medium-it-2 | ASR | 769M | 1e-5 | 8 | 5 | 500 steps | 0.01 | 49m per epoch | 94ms per sample |
| EdoAbati/whisper-large-v2-it | ASR | 1.5B | 1e-5 | 8 | 5 | 500 steps | 0.01 | 1h 17m per epoch | 238ms per sample |
| bert-base-multilingual-uncased | NLU | 167M | 5e-5 | 8 | 5 | 500 steps | 0.01 | 1m 22s per epoch | 1.5ms per sample |
| facebook/mbart-large-cc25 | NLU | 611M | 5e-5 | 8 | 5 | 500 steps | 0.01 | 7m 53s per epoch | 4.7ms per sample |
| dbmdz/bert-base-italian-xxl-uncased | NLU | 110M | 5e-5 | 8 | 5 | 500 steps | 0.01 | 1m 30s per epoch | 1.4ms per sample |
| morenolq/bart-it | NLU | 141M | 5e-5 | 8 | 5 | 500 steps | 0.01 | 1m 54s per epoch | 1.9 ms per sample |

In all cases, we opted for the AdamW optimizer. All experiments were run on a single NVIDIA A6000 GPU.


### SLU intent classification

The models used in the paper are available on the [Hugging Face Hub](https://huggingface.co/models).

- 🌍 [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- 🌍 [facebook/wav2vec2-xls-r-1b](https://huggingface.co/facebook/wav2vec2-xls-r-1b)
- 🇮🇹 [jonatasgrosman/wav2vec2-xls-r-1b-italian](https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-italian)
- 🇮🇹 [jonatasgrosman/wav2vec2-large-xlsr-53-italian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian)

### ASR

The models used in the paper are available on the [Hugging Face Hub](https://huggingface.co/models).

- 🌍 Whisper large (zero-shot ASR): [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)
- 🇮🇹 Whisper small: [ALM/whisper-it-small-augmented](https://huggingface.co/ALM/whisper-it-small-augmented)
- 🇮🇹 Whisper medium: [EdoAbati/whisper-medium-it-2](https://huggingface.co/EdoAbati/whisper-medium-it-2)
- 🇮🇹 Whisper large: [EdoAbati/whisper-large-v2-it](https://huggingface.co/EdoAbati/whisper-large-v2-it)

### NLU intent classification

The models used in the paper are available on the [Hugging Face Hub](https://huggingface.co/models).

- 🌍 [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased)
- 🌍 [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25)
- 🇮🇹 [dbmdz/bert-base-italian-xxl-uncased](https://huggingface.co/dbmdz/bert-base-italian-xxl-uncased)
- 🇮🇹 [morenolq/bart-it](https://huggingface.co/morenolq/bart-it)

## Citation

If you use this dataset in your research, please cite the following paper:

```
TO BE ADDED UPON PUBLICATION
```

## License

The dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

- [Paper describing the dataset and initial experiments](https://arxiv.org/abs/2306.08502)
- [Dataset on Zenodo](https://zenodo.org/record/8040649)
- [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by/4.0/)https://creativecommons.org/licenses/by/4.0/
