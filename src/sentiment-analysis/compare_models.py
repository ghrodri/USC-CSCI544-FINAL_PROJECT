import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import classes from training script
from train_bert_sentiment import FinancialSentimentDataset, DistilBERTSentimentClassifier

def evaluate_baseline_model(model, data_loader, device):
    """Evaluate untrained model with random predictions"""
    model.eval()
    predictions = []
    real_values = []
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating baseline'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(real_values, predictions)
    return accuracy, np.mean(losses), predictions, real_values

def plot_confusion_matrices(y_true_baseline, y_pred_baseline, y_true_trained, y_pred_trained):
    """Plot confusion matrices side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline confusion matrix
    cm_baseline = confusion_matrix(y_true_baseline, y_pred_baseline)
    im1 = ax1.imshow(cm_baseline, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im1, ax=ax1)
    ax1.set(xticks=np.arange(cm_baseline.shape[1]),
           yticks=np.arange(cm_baseline.shape[0]),
           xticklabels=['Negative', 'Neutral', 'Positive'],
           yticklabels=['Negative', 'Neutral', 'Positive'],
           title='Baseline Model (Untrained)',
           ylabel='Actual',
           xlabel='Predicted')
    
    # Add text annotations
    for i in range(cm_baseline.shape[0]):
        for j in range(cm_baseline.shape[1]):
            text = ax1.text(j, i, cm_baseline[i, j],
                           ha="center", va="center", color="white" if cm_baseline[i, j] > cm_baseline.max() / 2 else "black")
    
    # Trained confusion matrix
    cm_trained = confusion_matrix(y_true_trained, y_pred_trained)
    im2 = ax2.imshow(cm_trained, interpolation='nearest', cmap=plt.cm.Greens)
    ax2.figure.colorbar(im2, ax=ax2)
    ax2.set(xticks=np.arange(cm_trained.shape[1]),
           yticks=np.arange(cm_trained.shape[0]),
           xticklabels=['Negative', 'Neutral', 'Positive'],
           yticklabels=['Negative', 'Neutral', 'Positive'],
           title='Fine-tuned Model',
           ylabel='Actual',
           xlabel='Predicted')
    
    # Add text annotations
    for i in range(cm_trained.shape[0]):
        for j in range(cm_trained.shape[1]):
            text = ax2.text(j, i, cm_trained[i, j],
                           ha="center", va="center", color="white" if cm_trained[i, j] > cm_trained.max() / 2 else "black")
    
    plt.tight_layout()
    plt.savefig('./model_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 80)
    print("Financial Sentiment Analysis: Baseline vs Fine-tuned Model Comparison")
    print("=" * 80)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load test data
    data_dir = '../data/sentiment_splits'
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print(f"\nTest samples: {len(test_df)}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create test dataset
    test_dataset = FinancialSentimentDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_length=128
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    print("\n" + "-" * 60)
    print("BASELINE MODEL (Untrained DistilBERT)")
    print("-" * 60)
    
    # Initialize baseline model
    baseline_model = DistilBERTSentimentClassifier(n_classes=3)
    baseline_model = baseline_model.to(device)
    
    # Evaluate baseline
    baseline_acc, baseline_loss, baseline_preds, y_true = evaluate_baseline_model(
        baseline_model, test_loader, device
    )
    
    print(f"\nBaseline Test Accuracy: {baseline_acc:.4f}")
    print(f"Baseline Test Loss: {baseline_loss:.4f}")
    
    print("\nBaseline Classification Report:")
    print(classification_report(y_true, baseline_preds, 
                              target_names=['Negative', 'Neutral', 'Positive']))
    
    print("\n" + "-" * 60)
    print("FINE-TUNED MODEL")
    print("-" * 60)
    
    # Load fine-tuned model
    trained_model = DistilBERTSentimentClassifier(n_classes=3)
    
    # Try both possible model directories
    model_paths = ['./models/best_model.pt', './bert-models/best_model.pt']
    model_loaded = False
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            trained_model.load_state_dict(checkpoint['model_state_dict'])
            trained_model = trained_model.to(device)
            model_loaded = True
            print(f"\nLoaded fine-tuned model from: {model_path}")
            break
    
    if not model_loaded:
        print("\nError: Could not find trained model file!")
        return
    
    # Evaluate trained model
    trained_model.eval()
    trained_preds = []
    trained_losses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating fine-tuned model'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = trained_model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            trained_losses.append(loss.item())
            
            _, preds = torch.max(outputs, dim=1)
            trained_preds.extend(preds.cpu().numpy())
    
    trained_acc = accuracy_score(y_true, trained_preds)
    trained_loss = np.mean(trained_losses)
    
    print(f"\nFine-tuned Test Accuracy: {trained_acc:.4f}")
    print(f"Fine-tuned Test Loss: {trained_loss:.4f}")
    
    print("\nFine-tuned Classification Report:")
    print(classification_report(y_true, trained_preds, 
                              target_names=['Negative', 'Neutral', 'Positive']))
    
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    acc_improvement = trained_acc - baseline_acc
    acc_improvement_pct = (acc_improvement / baseline_acc) * 100
    
    print(f"\nAccuracy Improvement: {baseline_acc:.4f} → {trained_acc:.4f}")
    print(f"Absolute Improvement: +{acc_improvement:.4f}")
    print(f"Relative Improvement: +{acc_improvement_pct:.1f}%")
    
    print(f"\nLoss Reduction: {baseline_loss:.4f} → {trained_loss:.4f}")
    print(f"Loss Decreased by: {baseline_loss - trained_loss:.4f}")
    
    # Calculate per-class metrics
    baseline_report = classification_report(y_true, baseline_preds, output_dict=True)
    trained_report = classification_report(y_true, trained_preds, output_dict=True)
    
    print("\nPer-Class F1-Score Improvements:")
    for i, class_name in enumerate(['Negative', 'Neutral', 'Positive']):
        baseline_f1 = baseline_report[str(i)]['f1-score']
        trained_f1 = trained_report[str(i)]['f1-score']
        improvement = trained_f1 - baseline_f1
        print(f"{class_name}: {baseline_f1:.3f} → {trained_f1:.3f} (+{improvement:.3f})")
    
    # Plot confusion matrices
    plot_confusion_matrices(y_true, baseline_preds, y_true, trained_preds)
    print("\nConfusion matrices saved to: ./model_comparison_confusion_matrices.png")
    
    # Create performance comparison bar chart
    labels = ['Baseline', 'Fine-tuned']
    accuracy_scores = [baseline_acc, trained_acc]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, accuracy_scores, color=['lightcoral', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, score in zip(bars, accuracy_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.savefig('./model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Accuracy comparison chart saved to: ./model_accuracy_comparison.png")
    
    print("\n" + "=" * 60)
    print("Comparison complete!")

if __name__ == "__main__":
    main()