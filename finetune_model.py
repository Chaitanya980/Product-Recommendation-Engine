import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./pretrained_model')
model = BertForSequenceClassification.from_pretrained('./pretrained_model', num_labels=10)  # Assume 10 product classes for simplicity

dataset = load_dataset('parquet', data_files={'train': 'train.parquet', 'test': 'test.parquet'})

def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("target_product_class", "labels")  # Assume labels column

# Training arguments for fine-tuning
training_args = TrainingArguments(
    output_dir='./finetune_results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)


trainer.train()
model.save_pretrained('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')


baseline_acc = 0.60  
new_acc = trainer.evaluate()['eval_accuracy']
improvement = ((new_acc - baseline_acc) / baseline_acc) * 100
print(f"Achieved {improvement:.2f}% improvement in recommendation accuracy.")