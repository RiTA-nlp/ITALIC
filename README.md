# ITALIC: An ITALian Intent Classification Dataset

This repository contains the code and the dataset for the paper [ITALIC: An ITALian Intent Classification Dataset](#).

ITALIC is a new intent classification dataset for the Italian language, which is the first of its kind. It includes spoken and written utterances and is annotated with 60 intents. The dataset is available on the [Hugging Face Hub](#).

## Table of Contents

- [Data collection](#data-collection)
- [Dataset](#dataset)
- [Citation](#citation)
- [License](#license)

## Data collection

The data collection follows the MASSIVE NLU dataset which contains an annotated textual dataset for 60 intents. The data collection process is described in the paper [Massive Natural Language Understanding](https://arxiv.org/abs/2204.08582).

Following the MASSIVE NLU dataset, a pool of 70+ volunteers has been recruited to annotate the dataset. The volunteers were asked to record their voice while reading the utterances (the original text is available on MASSIVE dataset). Together with the audio, the volunteers were asked to provide a self-annotated description of the recording conditions (e.g., background noise, recording device). The audio recordings have also been validated and, in case of errors, re-recorded by the volunteers.

All the audio recordings included in the dataset have received a validation from at least two volunteers. All the audio recordings have been validated by native italian speakers (self-annotated).

## Dataset

The dataset is available on the [Hugging Face Hub](#). It is composed of 3 different splits:
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

## Citation

If you use this dataset in your research, please cite the following paper:

```
TO BE ADDED UPON PUBLICATION
```

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

## License

The dataset is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[1]: [Paper](#)
[2]: [Hugging Face Hub](#)
[3]: https://creativecommons.org/licenses/by-nc-sa/4.0/