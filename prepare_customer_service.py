"""
Prepare the Customer Service dataset for sentiment analysis.
Will save train.bin, val.bin containing the processed data, and meta.pkl containing the
encoder and decoder and other related info.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
import torch

def load_data():
    """Load the customer service datasets"""
    train_df = pd.read_csv('data/customer_service/train.csv')
    test_df = pd.read_csv('data/customer_service/test.csv')
    return train_df, test_df

def process_conversation(text):
    """Extract relevant parts from the conversation"""
    # Remove system messages like [Agent puts the customer on hold]
    lines = [line for line in text.split('\n') if not line.strip().startswith('[')]
    # Remove speaker labels (Agent: and Customer:)
    lines = [line.split(':', 1)[1].strip() if ':' in line else line.strip() for line in lines]
    # Join the cleaned lines
    return ' '.join(lines)

def prepare_data():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Add special tokens for sentiment
    special_tokens = {'pad_token': '<|pad|>',
                     'sentiment_tokens': ['<|positive|>', '<|negative|>', '<|neutral|>']}
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens['sentiment_tokens'],
                                'pad_token': special_tokens['pad_token']})

    # Load data
    train_df, test_df = load_data()
    
    # Process conversations
    train_df['processed_conversation'] = train_df['conversation'].apply(process_conversation)
    test_df['processed_conversation'] = test_df['conversation'].apply(process_conversation)

    # Create sentiment mapping
    sentiment_to_id = {'positive': 0, 'negative': 1, 'neutral': 2}
    
    # Convert sentiments to ids
    train_df['sentiment_id'] = train_df['customer_sentiment'].map(sentiment_to_id)
    test_df['sentiment_id'] = test_df['customer_sentiment'].map(sentiment_to_id)

    # Split training data into train and validation
    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['sentiment_id']
    )

    # Tokenize conversations
    def tokenize_and_pad(texts, max_length=512):
        return tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    # Process each split
    train_encodings = tokenize_and_pad(train_data['processed_conversation'])
    val_encodings = tokenize_and_pad(val_data['processed_conversation'])
    test_encodings = tokenize_and_pad(test_df['processed_conversation'])

    # Save processed data
    data_dir = 'data/customer_service'
    os.makedirs(data_dir, exist_ok=True)

    # Save tokenized data and labels
    torch.save({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': torch.tensor(train_data['sentiment_id'].values)
    }, os.path.join(data_dir, 'train.pt'))

    torch.save({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': torch.tensor(val_data['sentiment_id'].values)
    }, os.path.join(data_dir, 'val.pt'))

    torch.save({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': torch.tensor(test_df['sentiment_id'].values)
    }, os.path.join(data_dir, 'test.pt'))

    # Save metadata
    meta = {
        'vocab_size': len(tokenizer),
        'sentiment_to_id': sentiment_to_id,
        'id_to_sentiment': {v: k for k, v in sentiment_to_id.items()},
        'special_tokens': special_tokens,
        'max_length': 512,
        'num_classes': len(sentiment_to_id)
    }

    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # Print dataset statistics
    print(f"Vocabulary size: {len(tokenizer):,}")
    print(f"Train samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    print(f"Test samples: {len(test_df):,}")
    print("\nSentiment distribution:")
    print("Train:", train_data['customer_sentiment'].value_counts().to_dict())
    print("Val:", val_data['customer_sentiment'].value_counts().to_dict())
    print("Test:", test_df['customer_sentiment'].value_counts().to_dict())

if __name__ == '__main__':
    prepare_data()