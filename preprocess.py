import re
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

# Load the tokenizer
MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def clean_and_format_dialog(dialog_str):
    utterances = re.findall(r"'(.*?)'|\"(.*?)\"", str(dialog_str))
    utterances = [u[0].strip() or u[1].strip() for u in utterances if u[0] or u[1]]
    
    # Use the actual EOS token, not pad token
    eos_token = tokenizer.eos_token  # <|endoftext|>
    joined_dialog = eos_token.join(utterances) + eos_token
    return joined_dialog


def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['formatted_dialog'] = df['dialog'].apply(clean_and_format_dialog)
    dataset = Dataset.from_pandas(df[['formatted_dialog']])

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["formatted_dialog"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
        all_labels = []
        for input_ids in tokenized["input_ids"]:
            labels = input_ids.copy()
    
            # Only mask TRAILING padding, not EOS tokens mid-sequence
            # Walk from the end and mask pad tokens until we hit real content
            for i in range(len(labels) - 1, -1, -1):
                if labels[i] == tokenizer.pad_token_id:
                    labels[i] = -100
                else:
                    break  # stop at first real token from the right
    
            all_labels.append(labels)
    
        tokenized["labels"] = all_labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["formatted_dialog"])
    return tokenized_dataset