import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import hashlib
import re

def text_id(text: str) -> str:
    """Generate a unique ID for text"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = str(text).strip()
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text

def load_financial_phrasebank(file_path):
    """Load financial phrasebank text files"""
    data = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line and '@' in line:
                # Split at the last @ to get text and sentiment
                parts = line.rsplit('@', 1)
                if len(parts) == 2:
                    text, sentiment = parts
                    # Map sentiment labels
                    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
                    label = label_map.get(sentiment.strip().lower(), 1)
                    
                    data.append({
                        'text': clean_text(text),
                        'label': label,
                        'source': 'financial_phrasebank'
                    })
    
    return pd.DataFrame(data)

def load_aiera_parquet(file_path):
    """Load Aiera sentiment data from parquet"""
    df = pd.read_parquet(file_path)
    
    # Map columns
    data = []
    for _, row in df.iterrows():
        # Try different column names
        text = row.get('transcript', row.get('text', row.get('segment', '')))
        sentiment = row.get('sentiment', row.get('label', 'neutral'))
        
        # Map sentiment labels
        if isinstance(sentiment, str):
            label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
            label = label_map.get(sentiment.strip().lower(), 1)
        else:
            label = int(sentiment)
            
        if text:
            data.append({
                'text': clean_text(text),
                'label': label,
                'source': 'aiera'
            })
    
    return pd.DataFrame(data)

def main():
    # Paths
    raw_data_dir = "src/data/sentiment_raw_data"
    output_dir = "src/data/sentiment_splits"
    
    # Load all datasets
    all_data = []
    
    # 1. Load Financial Phrasebank files
    print("Loading Financial Phrasebank datasets...")
    
    # We'll use AllAgree for highest quality, but you can change this to include others
    phrasebank_files = ['Sentences_AllAgree.txt']
    # Uncomment below to include all agreement levels:
    # phrasebank_files = ['Sentences_AllAgree.txt', 'Sentences_75Agree.txt', 'Sentences_66Agree.txt', 'Sentences_50Agree.txt']
    
    for file in phrasebank_files:
        file_path = os.path.join(raw_data_dir, file)
        if os.path.exists(file_path):
            df = load_financial_phrasebank(file_path)
            print(f"  Loaded {len(df)} samples from {file}")
            all_data.append(df)
    
    # 2. Load Aiera dataset
    print("\nLoading Aiera dataset...")
    aiera_file = os.path.join(raw_data_dir, 'test-00000-of-00001.parquet')
    if os.path.exists(aiera_file):
        df = load_aiera_parquet(aiera_file)
        print(f"  Loaded {len(df)} samples from Aiera")
        all_data.append(df)
    
    # 3. Combine all data
    print("\nCombining datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total samples before deduplication: {len(combined_df)}")
    
    # 4. Remove duplicates based on text
    combined_df = combined_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"Total samples after deduplication: {len(combined_df)}")
    
    # 5. Add unique IDs
    combined_df['id'] = combined_df['text'].apply(text_id)
    
    # Handle potential ID collisions
    dup_ids = combined_df['id'].duplicated(keep=False)
    if dup_ids.any():
        for idx in combined_df[dup_ids].index:
            combined_df.at[idx, 'id'] = text_id(combined_df.at[idx, 'text'] + str(idx))
    
    # 6. Check label distribution
    print("\nLabel distribution:")
    label_counts = combined_df['label'].value_counts().sort_index()
    label_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
    for label, count in label_counts.items():
        print(f"  {label_names[label]}: {count} ({count/len(combined_df)*100:.1f}%)")
    
    # 7. Create train/val/test splits (70/15/15)
    print("\nCreating train/val/test splits...")
    
    # First split: train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=0.30, 
        stratify=combined_df['label'],
        random_state=42
    )
    
    # Second split: val (15%) vs test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df['label'],
        random_state=42
    )
    
    # 8. Save splits
    print(f"\nSaving splits to {output_dir}:")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files with proper column order
    columns = ['id', 'text', 'label', 'source']
    train_df[columns].to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df[columns].to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df[columns].to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # 9. Save detailed manifest
    manifest = {
        'dataset_info': {
            'total_samples': len(combined_df),
            'n_sources': len(combined_df['source'].unique()),
            'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'splits': {
            'train': {
                'n_samples': len(train_df),
                'label_distribution': train_df['label'].value_counts().sort_index().to_dict(),
                'source_distribution': train_df['source'].value_counts().to_dict()
            },
            'val': {
                'n_samples': len(val_df),
                'label_distribution': val_df['label'].value_counts().sort_index().to_dict(),
                'source_distribution': val_df['source'].value_counts().to_dict()
            },
            'test': {
                'n_samples': len(test_df),
                'label_distribution': test_df['label'].value_counts().sort_index().to_dict(),
                'source_distribution': test_df['source'].value_counts().to_dict()
            }
        },
        'sources': combined_df['source'].value_counts().to_dict(),
        'label_map': {0: 'negative', 1: 'neutral', 2: 'positive'},
        'split_ratio': '70/15/15',
        'random_seed': 42
    }
    
    with open(os.path.join(output_dir, 'dataset_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # 10. Print summary statistics
    print("\n" + "="*50)
    print("DATASET PREPARATION COMPLETE")
    print("="*50)
    print(f"\nTotal unique samples: {len(combined_df)}")
    print(f"Sources: {', '.join(combined_df['source'].unique())}")
    print("\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(combined_df)*100:.1f}%)")
    
    print("\nFiles created:")
    print(f"  - {output_dir}/train.csv")
    print(f"  - {output_dir}/val.csv")
    print(f"  - {output_dir}/test.csv")
    print(f"  - {output_dir}/dataset_manifest.json")
    
    # Also save manifest in sentiment-analysis folder for reference
    with open(os.path.join("src/sentiment-analysis", 'dataset_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()