import os
from transformers import DefaultDataCollator, AutoModelForCausalLM, TrainingArguments, Trainer
from preprocess import MODEL_NAME, tokenizer, prepare_dataset

OUTPUT_PATH = os.path.join('/home/majd/Desktop/', 'DialoGPT on DailyDialog')
# Load the model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.float()

data_collator = DefaultDataCollator()

# Preprocess dataset
train_dataset = prepare_dataset("data/DailyDialog/train.csv")
val_dataset = prepare_dataset("data/DailyDialog/validation.csv")

# Define training hyperparameters
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_PATH, 'DialoGPT'),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    save_strategy="steps",
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,
    warmup_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Start training
print("Starting fine-tuning...")
trainer.train()

# Save the final model and tokenizer
trainer.save_model(os.path.join(OUTPUT_PATH, 'DialoGPT-final'))
tokenizer.save_pretrained(os.path.join(OUTPUT_PATH, 'DialoGPT-final'))
print("Training complete and model saved.")