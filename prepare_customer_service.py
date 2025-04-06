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
import wandb  # Add wandb import

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
    # Initialize wandb
    wandb.init(
        project="customer-service-sentiment",
        name="data-preparation",
        config={
            "max_length": 512,
            "tokenizer": "gpt2",
            "val_split": 0.1,
            "random_seed": 42
        }
    )

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

    # Add conversation length for EDA
    train_df['conversation_length'] = train_df['processed_conversation'].str.split().str.len()
    test_df['conversation_length'] = test_df['processed_conversation'].str.split().str.len()

    # Perform Exploratory Data Analysis
    print("\n=== Exploratory Data Analysis ===")
    
    # 1. Dataset size information
    dataset_stats = {
        "total_train_samples": len(train_df),
        "total_test_samples": len(test_df),
    }
    wandb.log(dataset_stats)
    
    print("\nDataset sizes:")
    print(f"Total training samples: {len(train_df):,}")
    print(f"Total test samples: {len(test_df):,}")
    
    # 2. Sentiment distribution
    print("\nSentiment Distribution:")
    train_sentiment_dist = train_df['customer_sentiment'].value_counts(normalize=True) * 100
    
    # Log sentiment distribution to wandb
    wandb.log({
        "sentiment_distribution": wandb.Table(
            data=[[label, pct] for label, pct in train_sentiment_dist.items()],
            columns=["sentiment", "percentage"]
        )
    })
    
    print("\nTraining set sentiment distribution (%):")
    print(train_sentiment_dist)
    
    # 3. Conversation length statistics
    length_stats = train_df['conversation_length'].describe().to_dict()
    wandb.log({f"conversation_length/{k}": v for k, v in length_stats.items()})
    
    print("\nConversation Length Statistics:")
    print("\nTraining set conversation length:")
    print(train_df['conversation_length'].describe())
    
    # 4. Create and save distribution plots
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        data_dir = 'data/customer_service'
        os.makedirs(data_dir, exist_ok=True)
        
        # Sentiment distribution plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=train_df, x='customer_sentiment')
        plt.title('Distribution of Customer Sentiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'sentiment_distribution.png'))
        wandb.log({"sentiment_distribution_plot": wandb.Image(plt)})
        
        # Conversation length distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=train_df, x='conversation_length', bins=50)
        plt.title('Distribution of Conversation Lengths')
        plt.xlabel('Number of Words')
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'conversation_length_distribution.png'))
        wandb.log({"conversation_length_distribution": wandb.Image(plt)})
        
        plt.close('all')
    except ImportError:
        print("\nMatplotlib/Seaborn not available - skipping plots")

    # Create sentiment mapping
    sentiment_to_id = {'positive': 0, 'negative': 1, 'neutral': 2}
    
    # Convert sentiments to ids
    train_df['sentiment_id'] = train_df['customer_sentiment'].map(sentiment_to_id)
    test_df['sentiment_id'] = test_df['customer_sentiment'].map(sentiment_to_id)

    # Split training data into train and validation
    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['sentiment_id']
    )

    # Log split sizes
    split_stats = {
        "train_split_size": len(train_data),
        "val_split_size": len(val_data),
        "test_split_size": len(test_df)
    }
    wandb.log(split_stats)

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

    # Log tokenization stats
    token_stats = {
        "vocab_size": len(tokenizer),
        "max_sequence_length": 512,
        "train_sequences": len(train_encodings['input_ids']),
        "val_sequences": len(val_encodings['input_ids']),
        "test_sequences": len(test_encodings['input_ids'])
    }
    wandb.log(token_stats)

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

    # Log metadata to wandb
    wandb.log({"metadata": meta})

    # Print final dataset statistics
    print(f"\nFinal Dataset Statistics:")
    print(f"Vocabulary size: {len(tokenizer):,}")
    print(f"Train samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    print(f"Test samples: {len(test_df):,}")
    print("\nSentiment distribution:")
    print("Train:", train_data['customer_sentiment'].value_counts().to_dict())
    print("Val:", val_data['customer_sentiment'].value_counts().to_dict())
    print("Test:", test_df['customer_sentiment'].value_counts().to_dict())

    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    prepare_data()