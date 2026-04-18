# DialoGPT Chatbot — Fine-tuned on DailyDialog

A conversational AI chatbot built by fine-tuning Microsoft's [DialoGPT-small](https://huggingface.co/microsoft/DialoGPT-small) on the [DailyDialog](https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in) dataset. The model is trained to engage in natural conversations and supports multi-turn dialogue.

[![Kaggle Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/majdkhuzam/fine-tuning-dialogpt-on-dailydialog-dataset)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Model Configuration](#model-configuration)
- [Training Details](#training-details)

---

## Overview

This project fine-tunes DialoGPT-small — a pre-trained conversational language model by Microsoft — on the DailyDialog dataset, which contains thousands of high-quality, human-written dialogues covering a wide range of everyday topics. The result is a chatbot capable of generating contextually appropriate, multi-turn responses.

The pipeline consists of three stages:

1. **Preprocessing** — Dialogues are cleaned, formatted with EOS tokens as turn separators, tokenized, and padded to a fixed length.
2. **Training** — The model is fine-tuned using Hugging Face's `Trainer` API with standard causal language modeling.
3. **Inference** — A chat loop loads the fine-tuned model and maintains a running conversation history to produce coherent multi-turn responses.

---

## Project Structure

```
.
├── preprocess.py       # Data loading, cleaning, and tokenization
├── train.py            # Model fine-tuning using Hugging Face Trainer
├── inference.py        # Interactive chat loop using the fine-tuned model
├── requirements.txt    # Python dependencies
└── data/
    └── DailyDialog/
        ├── train.csv
        └── validation.csv
```

---

## Requirements

- Python 3.12+
- The packages listed in `requirements.txt`:

| Package        | Version      |
|----------------|--------------|
| `transformers` | 5.0.0        |
| `datasets`     | 4.8.3        |
| `pandas`       | 2.3.3        |
| `torch`        | 2.10.0+cu128 |
| `tensorflow`   | 2.19.0       |

---

## Setup

**1. Clone the repository:**

```bash
git clone https://github.com/MajdKhuzam/dialogpt-dailydialog-finetuning
cd dialogpt-dailydialog-finetuning
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Prepare the dataset:**

Place the DailyDialog CSV files in the expected directory:

```
data/DailyDialog/train.csv
data/DailyDialog/validation.csv
```

Each CSV file should contain a `dialog` column where each row holds a list of utterances representing a single conversation.

---

## Usage

### Training

Run the training script to fine-tune DialoGPT-small on the DailyDialog dataset:

```bash
python train.py
```

The fine-tuned model and tokenizer will be saved to:

```
~/Desktop/DialoGPT on DailyDialog/DialoGPT-final/
```

You can modify the `OUTPUT_PATH` variable in `train.py` to change the save location.

### Inference

Once training is complete, launch the interactive chat interface:

```bash
python inference.py
```

The chatbot will load the fine-tuned model and start an interactive session in your terminal. Type your messages and press Enter to receive a response. To end the conversation, type `quit`, `exit`, or `stop`.

```
>> You: Hello! How are you?
DialoGPT: I'm doing well, thanks for asking. How about you?
```

---

## Model Configuration

The following generation parameters are used during inference to produce natural, varied responses:

| Parameter              | Value | Description                                      |
|------------------------|-------|--------------------------------------------------|
| `max_new_tokens`       | 100   | Maximum number of tokens generated per response  |
| `no_repeat_ngram_size` | 3     | Prevents repetition of 3-gram phrases            |
| `do_sample`            | True  | Enables stochastic sampling                      |
| `top_k`                | 50    | Restricts sampling to the top 50 tokens          |
| `top_p`                | 0.95  | Nucleus sampling threshold                       |
| `temperature`          | 0.75  | Controls response creativity vs. focus           |

---

## Training Details

| Parameter                    | Value   |
|------------------------------|---------|
| Base model                   | `microsoft/DialoGPT-small` |
| Dataset                      | DailyDialog                |
| Epochs                       | 3                          |
| Train batch size             | 4                          |
| Eval batch size              | 4                          |
| Learning rate                | 2e-5                       |
| Weight decay                 | 0.01                       |
| Warmup steps                 | 500                        |
| Max sequence length          | 512 tokens                 |
| Mixed precision (`fp16`)     | Disabled                   |
| Evaluation & save strategy   | Every N steps              |
| Best model checkpoint        | Loaded at end of training  |
