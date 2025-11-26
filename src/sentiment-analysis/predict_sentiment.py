import torch
from train_roberta import RoBERTaSentimentClassifier
from transformers import RobertaTokenizer
import sys

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RoBERTaSentimentClassifier()      
model.load_state_dict(torch.load('src/sentiment-analysis/roberta_results/best_roberta.pt'))
model.to(device)
model.eval()

MAX_LENGTH = 128
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_text(text, tokenizer, max_length):
    text = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
        )
    
    return text


def predict_sentiment(text):
    global model, tokenizer, MAX_LENGTH
    
    # Tokenize text
    tokens = tokenize_text(text, tokenizer, MAX_LENGTH)

    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Predict sentiment
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = torch.argmax(output, dim=1).cpu().numpy()

    return prediction


if __name__ == "__main__":
    # Read input text
    input_path = sys.argv[1]

    with open(input_path, 'r') as f:
        text = f.read()

    # Predict sentiment
    prediction = predict_sentiment(text)

    print(prediction)