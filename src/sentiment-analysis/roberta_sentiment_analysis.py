import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW


class RoBERTaSentimentClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(RoBERTaSentimentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.roberta(input_ids, attention_mask)
        x = cls_emb = x.last_hidden_state[:, 0]
        x = self.dropout(cls_emb)

        return self.fc(x)


def tokenize_text(text, tokenizer, max_length):
    text = tokenizer(
        text.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
        )
    
    return text


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_correct = 0
    losses = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()
            losses.append(loss.item())
            
    accuracy = float(total_correct) / len(data_loader.dataset)
    average_loss = np.mean(losses)

    return accuracy, average_loss


def main():
    # ---- Import data ----
    data_directory = '../data/sentiment_splits'
    output_directory = './roberta_results'
    os.makedirs(output_directory, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_directory, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_directory, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_directory, 'test.csv'))

    # ---- Tokenize input text ----
    MAX_LENGTH = 64
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_text = tokenize_text(train_df['text'], tokenizer, MAX_LENGTH)
    test_text = tokenize_text(test_df['text'], tokenizer, MAX_LENGTH)
    val_text = tokenize_text(val_df['text'], tokenizer, MAX_LENGTH)

    # ---- Create dataset ----
    train_labels = torch.tensor(train_df['label'].values, dtype=torch.long)
    test_labels = torch.tensor(test_df['label'].values, dtype=torch.long)
    val_labels = torch.tensor(val_df['label'].values, dtype=torch.long)

    BATCH_SIZE = 8

    train_data = TensorDataset(train_text['input_ids'], train_text['attention_mask'], train_labels)
    test_data = TensorDataset(test_text['input_ids'], test_text['attention_mask'], test_labels)
    val_data = TensorDataset(val_text['input_ids'], val_text['attention_mask'], val_labels)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # ---- Initialize RoBERTa and learning parameters ----
    model = RoBERTaSentimentClassifier(num_classes=3)

    EPOCHS = 3
    LEARNING_RATE = 2e-5

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )


    # ---- Training loop ----
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch + 1}')

        model.train()
        total_correct = 0
        losses = []

        for epoch in range(EPOCHS):
            print(f'Epoch: {epoch + 1}')

            model.train()
            total_correct = 0
            losses = []

            for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)
                
                losses.append(loss.item())
                preds = torch.argmax(output, dim=1)
                total_correct += (preds == labels).sum().item()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                    print(f"  [Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

            train_acc = float(total_correct) / len(train_loader.dataset)

            # Save bast model
            val_acc, val_loss = evaluate_model(model, val_loader, criterion)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(output_directory, 'best_roberta.pt'))


    # ---- Evaluate model ----
    model.load_state_dict(torch.load(os.path.join(output_directory, 'best_roberta.pt')))
    test_acc, test_loss = evaluate_model(model, test_loader, criterion)

    print(f'Training Accuracy: {train_acc}')
    print(f'Validation Accuracy: {val_acc}')
    print(f'Testing Accuracy: {test_acc}')

    model.eval()
    test_preds, test_true = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels.numpy())


    results_path = os.path.join(output_directory, 'roberta_results.txt')
    report = classification_report(test_true, test_preds, target_names=['Negative', 'Neutral', 'Positive'])
    cm = confusion_matrix(test_true, test_preds)

    with open(results_path, 'w') as f:
        f.write(f'Training Accuracy: {train_acc:.4f}\n')
        f.write(f'Validation Accuracy: {val_acc:.4f}\n')
        f.write(f'Testing Accuracy: {test_acc:.4f}\n\n')

        f.write('Classification Report:\n')
        f.write(report)
        f.write('\n\nConfusion Matrix:\n')
        f.write(np.array2string(cm))
        


if __name__ == "__main__":
    main()

