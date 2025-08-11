
import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load preprocessed data 
dataset = load_dataset('parquet', data_files={'train': 'train.parquet'})

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')


training_args = TrainingArguments(
    output_dir='./pretrain_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()
model.save_pretrained('./pretrained_model')
tokenizer.save_pretrained('./pretrained_model')
