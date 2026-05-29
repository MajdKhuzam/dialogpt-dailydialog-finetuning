# DialoGPT Chatbot — Fine-tuned on DailyDialog

A conversational AI chatbot built by fine-tuning Microsoft's [DialoGPT-small](https://huggingface.co/microsoft/DialoGPT-small) on the [DailyDialog](https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in) dataset. The model is trained to engage in natural conversations and supports multi-turn dialogue. Comes with an interactive terminal chat loop and a web UI.

[![Kaggle Notebook](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/majdkhuzam/fine-tuning-dialogpt-on-dailydialog-dataset)

---

## Overview

This project fine-tunes DialoGPT-small — a pre-trained conversational language model by Microsoft — on the DailyDialog dataset, which contains thousands of high-quality, human-written dialogues covering a wide range of everyday topics. The result is a chatbot capable of generating contextually appropriate, multi-turn responses.

The pipeline consists of three stages:

1. **Preprocessing** — Dialogues are cleaned, formatted with EOS tokens as turn separators, tokenized, and padded to a fixed length.
2. **Training** — The model is fine-tuned using Hugging Face's `Trainer` API with standard causal language modeling.
3. **Inference** — A FastAPI web server loads the fine-tuned model and exposes a REST API for multi-turn conversation, with a built-in static web UI for interacting with the chatbot directly in the browser.

---

## Project Structure

```
.
├── preprocess.py       # Load, clean, tokenize DailyDialog CSV files
├── train.py            # Fine-tune DialoGPT-small with Hugging Face Trainer
├── inference.py        # Interactive terminal chat loop
├── app.py              # FastAPI web server (REST API + static frontend)
├── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html      # Retro-styled chat UI (single-page app)
├── data/
│   └── DailyDialog/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
└── output/
    ├── DialoGPT/       # Training checkpoints
    └── DialoGPT-final/ # Saved fine-tuned model and tokenizer
```

---

## Requirements

- Python 3.12+
- Dependencies listed in `requirements.txt`

| Package        | Version      |
|----------------|--------------|
| `transformers` | 5.0.0        |
| `datasets`     | 4.8.3        |
| `pandas`       | 2.3.3        |
| `torch`        | 2.10.0+cu128 |
| `fastapi`      | 0.136.3      |
| `uvicorn`      | 0.48.0       |

---

## Setup

```bash
git clone https://github.com/MajdKhuzam/dialogpt-dailydialog-finetuning
cd dialogpt-dailydialog-finetuning
pip install -r requirements.txt
```

Place the DailyDialog CSV files in `data/DailyDialog/`. Each CSV must contain a `dialog` column where each row is a list of utterances (a single conversation).

---

## Usage

### 1. Preprocess & Train

```bash
python train.py
```

This runs preprocessing and fine-tuning. The model is saved to `output/DialoGPT-final/`.

### 2. Terminal Chat

```bash
python inference.py
```

Type messages interactively. Exit with `quit`, `exit`, or `stop`.

### 3. Web UI

```bash
python app.py
```

Open `http://localhost:8000` in a browser. The FastAPI server serves the chat frontend and exposes two endpoints:

| Endpoint       | Method | Description                         |
|----------------|--------|-------------------------------------|
| `/chat`        | POST   | Send a message (requires `session_id` and `message`) |
| `/new_session` | GET    | Get a fresh `session_id`            |

The frontend manages session IDs automatically.

---

## Generation Parameters

| Parameter              | Value | Description                                      |
|------------------------|-------|--------------------------------------------------|
| `max_new_tokens`       | 100   | Max tokens generated per response                |
| `no_repeat_ngram_size` | 3     | Prevents repetition of 3-gram phrases            |
| `do_sample`            | True  | Enables stochastic sampling                      |
| `top_k`                | 50    | Restricts sampling to top 50 tokens              |
| `top_p`                | 0.95  | Nucleus sampling threshold                       |
| `temperature`          | 0.75  | Controls creativity vs. focus                    |

---

## Training Details

| Parameter                    | Value                      |
|------------------------------|----------------------------|
| Base model                   | `microsoft/DialoGPT-small` |
| Dataset                      | DailyDialog                |
| Epochs                       | 3                          |
| Train batch size             | 4                          |
| Learning rate                | 2e-5                       |
| Weight decay                 | 0.01                       |
| Warmup steps                 | 500                        |
| Max sequence length          | 512 tokens                 |
| Mixed precision (`fp16`)     | Disabled                   |
| Evaluation & save strategy   | Every N steps              |
| Best model checkpoint        | Loaded at end of training  |
