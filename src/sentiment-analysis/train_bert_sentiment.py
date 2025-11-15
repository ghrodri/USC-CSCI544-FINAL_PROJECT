import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from tqdm import tqdm
import json
from datetime import datetime

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class DistilBERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super(DistilBERTSentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.dropout(pooled_output)
        return self.classifier(output)

def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    return train_df, val_df, test_df

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': np.mean(losses[-100:])})
    
    return correct_predictions.float() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.float() / len(data_loader.dataset), np.mean(losses)

def get_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Getting predictions'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    
    return predictions, real_values

def main():
    print("Starting Financial Sentiment Analysis Training with DistilBERT")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "mps":
        print("Mac GPU (MPS) detected and will be used for training!")
    
    data_dir = '../data/sentiment_splits'
    output_dir = './bert-models'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading data...")
    train_df, val_df, test_df = load_data(data_dir)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    print("\nCreating datasets...")
    train_dataset = FinancialSentimentDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    val_dataset = FinancialSentimentDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    test_dataset = FinancialSentimentDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model = DistilBERTSentimentClassifier(n_classes=3)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }
    
    best_val_acc = 0
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc, val_loss = eval_model(model, val_loader, device)
        
        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)
        
        print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
        print(f'Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, test_loss = eval_model(model, test_loader, device)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')
    
    y_pred, y_true = get_predictions(model, test_loader, device)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    results = {
        'model': 'DistilBERT',
        'test_accuracy': test_acc.item(),
        'test_loss': test_loss,
        'training_history': history,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'max_length': MAX_LENGTH,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print("Training complete!")

if __name__ == "__main__":
    main()