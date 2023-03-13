# xlm-roberta-base-lora-language-detection

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on the [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information) dataset. Using the [PEFT-LoRA](https://github.com/huggingface/peft/) method to only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs.

## Model description

This model is an XLM-RoBERTa transformer model with a classification head on top (i.e. a linear layer on top of the pooled output). 
For additional information please refer to the [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) model card or to the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Conneau et al.

## Intended uses & limitations

You can directly use this model as a language detector, i.e. for sequence classification tasks. Currently, it supports the following 20 languages: 

`arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr), hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru), swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh)`

## Training and evaluation data

The model was fine-tuned on the [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information) dataset, which consists of text sequences in 20 languages. The training set contains 70k samples, while the validation and test sets 10k each. The average accuracy on the test set is **99.4%** (this matches the average macro/weighted F1-score being the test set perfectly balanced). A more detailed evaluation is provided by the following table.

| Language | Precision | Recall | F1-score | support |
|:--------:|:---------:|:------:|:--------:|:-------:|
|ar        | 1.000     |0.998   |0.999     |  500    |
|bg        | 0.992     |1.000   |0.996     |  500    |
|de        | 1.000     |1.000   |1.000     |  500    |
|el        | 1.000     |1.000   |1.000     |  500    |
|en        | 0.992     |0.992   |0.992     |  500    |
|es        | 0.994     |0.992   |0.993     |  500    |
|fr        | 0.998     |0.998   |0.998     |  500    |
|hi        | 0.945     |1.000   |0.972     |  500    |
|it        | 1.000     |0.984   |0.992     |  500    |
|ja        | 1.000     |1.000   |1.000     |  500    |
|nl        | 0.996     |0.992   |0.994     |  500    |
|pl        | 0.992     |0.988   |0.990     |  500    |
|pt        | 0.988     |0.986   |0.987     |  500    |
|ru        | 0.998     |0.996   |0.997     |  500    |
|sw        | 0.992     |0.994   |0.993     |  500    |
|th        | 1.000     |1.000   |1.000     |  500    |
|tr        | 1.000     |1.000   |1.000     |  500    |
|ur        | 1.000     |0.964   |0.982     |  500    |
|vi        | 1.000     |1.000   |1.000     |  500    |
|zh        | 1.000     |1.000   |1.000     |  500    |

### Benchmarks

As a baseline to compare `xlm-roberta-base-lora-language-detection` against, we have used the Python [langid](https://github.com/saffsd/langid.py) library. Since it comes pre-trained on 97 languages, we have used its `.set_languages()` method to constrain the language set to our 20 languages. The average accuracy of langid on the test set is **98.5%**. More details are provided by the table below.

| Language | Precision | Recall | F1-score | support |
|:--------:|:---------:|:------:|:--------:|:-------:|
|ar        |0.990      |0.970   |0.980     |500      |
|bg        |0.998      |0.964   |0.981     |500      |
|de        |0.992      |0.944   |0.967     |500      |
|el        |1.000      |0.998   |0.999     |500      |
|en        |1.000      |1.000   |1.000     |500      |
|es        |1.000      |0.968   |0.984     |500      |
|fr        |0.996      |1.000   |0.998     |500      |
|hi        |0.949      |0.976   |0.963     |500      |
|it        |0.990      |0.980   |0.985     |500      |
|ja        |0.927      |0.988   |0.956     |500      |
|nl        |0.980      |1.000   |0.990     |500      |
|pl        |0.986      |0.996   |0.991     |500      |
|pt        |0.950      |0.996   |0.973     |500      |
|ru        |0.996      |0.974   |0.985     |500      |
|sw        |1.000      |1.000   |1.000     |500      |
|th        |1.000      |0.996   |0.998     |500      |
|tr        |0.990      |0.968   |0.979     |500      |
|ur        |0.998      |0.996   |0.997     |500      |
|vi        |0.971      |0.990   |0.980     |500      |
|zh        |1.000      |1.000   |1.000     |500      |

## Using the model for inference

```python
# pip install -q loralib transformers
# pip install -q git+https://github.com/huggingface/peft.git@main

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

peft_model_id = "dominguesm/xlm-roberta-base-lora-language-detection"

# Load the Peft model config
peft_config = PeftConfig.from_pretrained(peft_model_id)

# Load the base model config
base_config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)

# Load the base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path, config=base_config
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the inference model
inference_model = PeftModel.from_pretrained(base_model, peft_model_id)

# Load the pipeline
pipe = pipeline("text-classification", model=inference_model, tokenizer=tokenizer)


def detect_lang(text: str) -> str:
    # This code runs on CPU, so we use torch.cpu.amp.autocast to perform
    # automatic mixed precision.
    with torch.cpu.amp.autocast():
        # or `with torch.cuda.amp.autocast():`
        pred = pipe(text)
    return pred


detect_lang(
    "Cada qual sabe amar a seu modo; o modo, pouco importa; o essencial é que saiba amar."
)
# [{'label': 'pt', 'score': 0.9959434866905212}]

```

### Framework versions

- torch 1.13.1+cu116
- datasets 2.10.1
- sklearn 1.2.1
- transformers 4.27.0.dev0
- langid 1.1.6
- peft 0.3.0.dev0

## Note

This study was fully based and inspired by the [xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection) model, developed by [Luca Papariello](https://github.com/LucaPapariello).