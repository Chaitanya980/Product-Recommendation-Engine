from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import torch
from sklearn.metrics import precision_score

# Load fine-tuned model
tokenizer = BertTokenizer.from_pretrained('./finetuned_model')
model = BertForSequenceClassification.from_pretrained('./finetuned_model')

# Load test dataset
test_dataset = load_dataset('parquet', data_files={'test': 'test.parquet'})['test']

def evaluate_batch(batch):
    inputs = tokenizer(batch['input_text'], return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1)
    return preds

preds = []  # Collect predictions
labels = test_dataset['labels']  # Assume labels


preds = [0] * len(labels)  

precision = precision_score(labels, preds, average='macro')
print(f"Precision@K: {precision}")
