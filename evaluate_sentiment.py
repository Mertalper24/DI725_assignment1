"""
Evaluation script for sentiment analysis model.
Loads a trained model and evaluates it on the test set with detailed metrics and analysis.
"""
import os
import torch
import pickle
import tiktoken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from model import GPTConfig, SentimentGPT

# Create a custom ModelLoader to handle different model architectures
class ModelLoader:
    @staticmethod
    def load_model(checkpoint_path):
        """Load model from checkpoint, handling different architectures"""
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
        
        # Determine model type based on the checkpoint
        model_name = "unknown"
        if "out-sentiment-ft" in checkpoint_path:
            model_name = "fine-tuned"
        else:
            model_name = "trained-from-scratch"
        
        # Create model based on checkpoint arguments
        gptconf = GPTConfig(**checkpoint['model_args'])
        
        # Try to load with the regular model first
        try:
            model = SentimentGPT(gptconf)
            model.load_state_dict(checkpoint['model'])
            return model, model_name
        except RuntimeError as e:
            print(f"Architecture mismatch detected: {str(e)}")
            print("Attempting to load with compatible model architecture...")
            raise RuntimeError("Could not load model with compatible architecture")

def predict_sentiment(model, text, encoder, device='cuda', max_length=512):
    # Encode text and truncate if needed
    encoded = encoder.encode(text)[:max_length]  # Truncate to max_length
    x = torch.tensor([encoded], dtype=torch.long, device=device)
    
    # Get prediction without applying additional calibration
    with torch.no_grad():
        # Let the model's forward method handle calibration
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
    
    return pred.item(), probs[0].tolist()

def load_test_data():
    """Load the test data from the validation set file"""
    with open('data/customer_service/val.bin', 'rb') as f:
        data = pickle.load(f)
    
    texts = data['texts']
    labels = data['labels']
    
    print(f"Loaded {len(texts)} test examples")
    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"Class distribution in test set: {distribution}")
    
    return texts, labels

def evaluate_model(model, texts, labels, encoder, device='cuda'):
    """Evaluate the model on the test set and return metrics"""
    y_true = []
    y_pred = []
    confidence_scores = []
    
    print(f"Evaluating model on {len(texts)} test examples...")
    
    # Process in batches to avoid memory issues
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        for text, label in zip(batch_texts, batch_labels):
            # Skip very short texts
            if len(text) < 10:
                continue
                
            # Convert tokens to text and truncate if needed
            token_text = encoder.decode(text[:512])  # Truncate to model's block size
            pred, probs = predict_sentiment(model, token_text, encoder, device)
            
            y_true.append(label)
            y_pred.append(pred)
            confidence_scores.append(probs)
            
        # Print progress
        if (i + batch_size) % 100 == 0:
            print(f"Processed {i + batch_size}/{len(texts)} examples")
    
    return y_true, y_pred, confidence_scores

def calculate_metrics(y_true, y_pred):
    """Calculate various evaluation metrics"""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Class names for reporting
    class_names = ['positive', 'neutral', 'negative']
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # F1 scores
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Precision and recall
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    metrics = {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot and save confusion matrix as image"""
    # Create plots directory in the model output directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    # Labels, title and ticks
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()

def plot_confidence_distribution(confidence_scores, y_true, y_pred, output_dir):
    """Plot confidence score distributions for correct vs incorrect predictions"""
    # Create plots directory in the model output directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    confidence_scores = np.array(confidence_scores)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get max confidence for each prediction
    max_confidence = np.max(confidence_scores, axis=1)
    
    # Split into correct and incorrect predictions
    correct = y_true == y_pred
    
    plt.figure(figsize=(10, 6))
    plt.hist([max_confidence[correct], max_confidence[~correct]], 
             bins=20, alpha=0.7, label=['Correct predictions', 'Incorrect predictions'])
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores for Correct vs Incorrect Predictions')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'confidence_distribution.png'))
    plt.close()
    
    # Also plot confidence by true class
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(['positive', 'neutral', 'negative']):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:  # Check if we have examples of this class
            plt.hist(max_confidence[class_mask], bins=15, alpha=0.7, label=f'True {class_name}')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Score Distribution by True Class')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'confidence_by_class.png'))
    plt.close()


def main():
    print("Loading model...")
    # Try to load model from specified path or use fallback
    ckpt_path = None
    
    # Check command line arguments for model path
    import sys
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
        print(f"Using checkpoint path from command line: {ckpt_path}")
    
    # Try fine-tuned model first, then regular model
    if ckpt_path is None:
        try:
            ckpt_path = 'out-sentiment-ft/best_ckpt.pt'
            model, model_name = ModelLoader.load_model(ckpt_path)
        except FileNotFoundError:
            try:
                ckpt_path = 'out-sentiment/best_ckpt.pt'
                model, model_name = ModelLoader.load_model(ckpt_path)
            except FileNotFoundError:
                raise FileNotFoundError("No checkpoint found! Make sure you've trained a model first.")
    else:
        model, model_name = ModelLoader.load_model(ckpt_path)
    
    print(f"Successfully loaded {model_name} model from {ckpt_path}")
    
    model.eval()
    model.cuda()
    
    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Load test data
    texts, labels = load_test_data()
    
    # Evaluate model
    y_true, y_pred, confidence_scores = evaluate_model(model, texts, labels, enc)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print results
    print("\n" + "="*50)
    print(f"SENTIMENT ANALYSIS EVALUATION RESULTS ({model_name})")
    print("="*50)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (micro): {metrics['f1_micro']:.4f}")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nF1 Scores per class:")
    for i, score in enumerate(metrics['f1_per_class']):
        class_name = ['positive', 'neutral', 'negative'][i]
        print(f"  {class_name}: {score:.4f}")
    
    print(f"\nPrecision (weighted): {metrics['precision']:.4f}")
    print(f"Recall (weighted): {metrics['recall']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Get the output directory from the checkpoint path
    output_dir = os.path.dirname(ckpt_path)
    
    # Plot confusion matrix
    class_names = ['positive', 'neutral', 'negative']
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, output_dir)
    print(f"Confusion matrix saved as '{output_dir}/plots/confusion_matrix.png'")
    
    # Plot confidence distribution
    plot_confidence_distribution(confidence_scores, y_true, y_pred, output_dir)
    print(f"Confidence distribution plots saved in '{output_dir}/plots' directory")
    
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"SENTIMENT ANALYSIS EVALUATION RESULTS ({model_name})\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score (micro): {metrics['f1_micro']:.4f}\n")
        f.write(f"F1 Score (macro): {metrics['f1_macro']:.4f}\n")
        f.write(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}\n\n")
        
        f.write("F1 Scores per class:\n")
        for i, score in enumerate(metrics['f1_per_class']):
            class_name = ['positive', 'neutral', 'negative'][i]
            f.write(f"  {class_name}: {score:.4f}\n")
        
        f.write(f"\nPrecision (weighted): {metrics['precision']:.4f}\n")
        f.write(f"Recall (weighted): {metrics['recall']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'] + "\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(metrics['confusion_matrix']) + "\n\n")
        
        f.write("\nReasons for Evaluation Metric Choice:\n")
        f.write("1. Accuracy: Simple and intuitive, but not ideal for imbalanced classes.\n")
        f.write("2. F1-Score: Balances precision and recall, critical for imbalanced data.\n")
        f.write("   - Macro: Equal weight to each class regardless of frequency.\n")
        f.write("   - Weighted: Takes class imbalance into account.\n")
        f.write("3. Confusion Matrix: Visualizes where the model makes mistakes.\n")
        f.write("4. Per-class metrics: Essential for understanding performance on minority classes.\n")
    
    print("\nEvaluation complete. Results saved to 'evaluation_metrics.txt'")

if __name__ == "__main__":
    main() 