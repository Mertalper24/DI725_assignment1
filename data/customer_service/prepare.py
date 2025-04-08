"""
Prepare the customer service dataset for sentiment analysis.
Will save train.bin, val.bin containing the encoded text and labels,
and meta.pkl containing the encoder/decoder and other metadata.
Includes Exploratory Data Analysis (EDA) and visualizations.
"""
import os
import csv
import pickle
import numpy as np
import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

def load_data_with_analysis(filename):
    """Load data and perform exploratory data analysis"""
    # Load full dataset as pandas DataFrame for analysis
    df = pd.read_csv(filename)
    
    # EDA Part 1: Basic Dataset Information
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print("\nColumns in the dataset:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nMissing values by column:")
    print(df.isnull().sum())
    
    # EDA Part 2: Sentiment Distribution
    print("\nSentiment distribution:")
    sentiment_counts = df['customer_sentiment'].value_counts()
    print(sentiment_counts)
    
    # Visualize sentiment distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='customer_sentiment', data=df)
    plt.title('Distribution of Customer Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/sentiment_distribution.png')
    plt.close()
    
    # EDA Part 3: Correlation with other features
    print("\nExploring correlations between sentiment and other features:")
    
    # Convert sentiment to numeric for correlation analysis
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    df['sentiment_numeric'] = df['customer_sentiment'].map(sentiment_map)
    
    # Check correlation with relevant features
    features_to_check = ['issue_complexity', 'agent_experience_level']
    for feature in features_to_check:
        if feature in df.columns:
            print(f"\nRelationship between {feature} and sentiment:")
            
            if df[feature].dtype == 'object':  # Categorical feature
                cross_tab = pd.crosstab(df[feature], df['customer_sentiment'])
                print(cross_tab)
                
                # Visualize relationship
                plt.figure(figsize=(12, 7))
                cross_tab_percent = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
                cross_tab_percent.plot(kind='bar', stacked=True)
                plt.title(f'Sentiment Distribution by {feature}')
                plt.xlabel(feature)
                plt.ylabel('Percentage')
                plt.xticks(rotation=45)
                plt.legend(title='Sentiment')
                plt.tight_layout()
                plt.savefig(f'plots/sentiment_by_{feature}.png')
                plt.close()
    
    # EDA Part 4: Conversation Length Analysis
    df['conversation_length'] = df['conversation'].str.len()
    print("\nConversation length statistics:")
    print(df['conversation_length'].describe())
    
    # Visualize conversation length by sentiment
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='customer_sentiment', y='conversation_length', data=df)
    plt.title('Conversation Length by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Conversation Length (characters)')
    plt.savefig('plots/conversation_length_by_sentiment.png')
    plt.close()
    
    # Justification for our approaches
    print("\n=== DATA PRE-PROCESSING APPROACH ===")
    print("1. We're using only the conversation text and sentiment labels, omitting other columns as specified.")
    print("2. We're applying the following pre-processing steps:")
    print("   - Removing conversations with missing data")
    print("   - Tokenizing text using GPT-2 tokenizer for compatibility with the model")
    print("   - Handling very short conversations (< 10 tokens) by removing them")
    print("   - Truncating very long conversations to 1024 tokens to fit model constraints")
    print("3. We're creating a stratified train/val split to maintain class distribution")
    print("4. We're applying balanced sampling during training to address class imbalance")
    
    # Process the data for model training
    texts = []
    labels = []
    
    # Expected column names
    SENTIMENT_COL = 'customer_sentiment'
    CONVERSATION_COL = 'conversation'  # The column containing the conversation
    
    # Check if the conversation column exists
    if CONVERSATION_COL not in df.columns:
        CONVERSATION_COL = df.columns[-1]  # Use the last column if not found
        print(f"Using {CONVERSATION_COL} for conversation text")
    
    # Process each row
    for i, row in df.iterrows():
        try:
            # Get the conversation text and sentiment
            text = str(row[CONVERSATION_COL]).strip()
            sentiment = str(row[SENTIMENT_COL]).lower().strip()
            
            # Skip any rows with missing data
            if not text or not sentiment or sentiment not in sentiment_map:
                continue
            
            # Encode text
            encoded = enc.encode(text)
            
            # Skip if text is too short
            if len(encoded) < 10:
                continue
                
            # Truncate if text is too long
            if len(encoded) > 1024:
                encoded = encoded[:1024]
            
            texts.append(encoded)
            labels.append(sentiment_map[sentiment])
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue
    
    return texts, labels, df

# Load training data and perform EDA
train_file = os.path.join(os.path.dirname(__file__), 'train.csv')
print("Loading training data from:", train_file)
texts, labels, df = load_data_with_analysis(train_file)

# Print tokenization statistics
token_lengths = [len(t) for t in texts]
print(f"\n=== TOKENIZATION STATISTICS ===")
print(f"Total samples after preprocessing: {len(texts)}")
print(f"Average sequence length: {np.mean(token_lengths):.1f} tokens")
print(f"Median sequence length: {np.median(token_lengths):.1f} tokens")
print(f"Min sequence length: {min(token_lengths)} tokens")
print(f"Max sequence length: {max(token_lengths)} tokens")

# Plot token length distribution
plt.figure(figsize=(12, 6))
plt.hist(token_lengths, bins=50)
plt.title('Distribution of Conversation Lengths (in tokens)')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.savefig('plots/token_length_distribution.png')
plt.close()

# Sentiment distribution after preprocessing
sentiment_counts = {}
for label in labels:
    sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
print("\n=== SENTIMENT DISTRIBUTION AFTER PREPROCESSING ===")
for sentiment, count in sentiment_counts.items():
    print(f"Class {sentiment}: {count} samples ({count/len(labels)*100:.2f}%)")

# Split into train/val (90/10 split)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)

# Convert to numpy arrays
train_data = {'texts': train_texts, 'labels': train_labels}
val_data = {'texts': val_texts, 'labels': val_labels}

# Save train and val data
with open(os.path.join(os.path.dirname(__file__), 'train.bin'), 'wb') as f:
    pickle.dump(train_data, f)
with open(os.path.join(os.path.dirname(__file__), 'val.bin'), 'wb') as f:
    pickle.dump(val_data, f)

# Generate example token sequences for each sentiment class
class_examples = {0: [], 1: [], 2: []}
for text, label in zip(texts[:1000], labels[:1000]):  # Check first 1000 samples
    if len(class_examples[label]) < 2:  # Store 2 examples per class
        # Decode to show actual text
        decoded = enc.decode(text[:50])  # First 50 tokens
        class_examples[label].append(decoded + "...")

# Save meta information
meta = {
    'vocab_size': enc.n_vocab,
    'sentiment_classes': 3,
    'sentiment_map': {'0': 'positive', '1': 'neutral', '2': 'negative'},
    'reverse_sentiment_map': {0: 'positive', 1: 'neutral', 2: 'negative'},
    'tokenizer': 'gpt2',
    'max_seq_length': 1024,
    'class_distribution': sentiment_counts,
    'class_examples': class_examples,
    'token_length_stats': {
        'mean': float(np.mean(token_lengths)),
        'median': float(np.median(token_lengths)),
        'min': min(token_lengths),
        'max': max(token_lengths)
    }
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\n=== DATASET SPLITTING ===")
print(f"Train samples: {len(train_texts)}")
print(f"Val samples: {len(val_texts)}")

print("\nProcessing complete! Visualizations saved to 'plots/' directory.")