import torch
import pickle
import tiktoken
from model import GPTConfig, SentimentGPT

def predict_sentiment(model, text, encoder, device='cuda'):
    # Encode text
    encoded = encoder.encode(text)
    x = torch.tensor([encoded], dtype=torch.long, device=device)
    
    # Get prediction with slight calibration for balance
    with torch.no_grad():
        logits, _ = model(x)
        
        # Slight calibration to balance predictions
        # Just enough to offset the observed bias but not too much
        calibration = torch.tensor([0.8, 0.6, -0.7], device=device)
        logits = logits + calibration
        
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
    
    return pred.item(), probs[0].tolist()

# Load model
ckpt_path = 'out-sentiment/best_ckpt.pt'
checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=True)
gptconf = GPTConfig(**checkpoint['model_args'])
model = SentimentGPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Test examples
test_examples = [
    {
        "text": """
        Customer: Hi, I just wanted to say that your support team has been amazing. The issue was resolved quickly and efficiently.
        Agent: Thank you for your kind words! We're glad we could help. Is there anything else you need assistance with?
        Customer: No, that's all. Thank you for the great service!
        """,
        "expected": "positive"
    },
    {
        "text": """
        Customer: I need to verify my email address for login.
        Agent: I'll help you with that. Could you please confirm the email address registered with your account?
        Customer: It's customer@email.com
        Agent: Thank you. I've sent a verification link to your email. Please check and click on it.
        Customer: Ok, I'll check.
        """,
        "expected": "neutral"
    },
    {
        "text": """
        Customer: I've been trying to log in for hours and it's not working! This is ridiculous!
        Agent: I apologize for the inconvenience. Let me help you resolve this issue.
        Customer: I've already tried everything you're going to suggest. This is a waste of time!
        """,
        "expected": "negative"
    },
    {
        "text": """
        Customer: I'm having trouble resetting my password.
        Agent: I understand. Let me guide you through the reset process.
        Customer: Thanks, I appreciate the help.
        Agent: You're welcome! Is there anything else I can assist you with today?
        Customer: No, that's all. Have a nice day.
        """,
        "expected": "neutral"
    },
    {
        "text": """
        Customer: This is the third time I've called about this issue and nobody can fix it!
        Agent: I apologize for the frustration. Let me see what I can do to resolve this.
        Customer: I don't think you'll be able to help either. Your company's service is terrible.
        """,
        "expected": "negative"
    }
]

print("Testing model with example conversations...\n")
sentiment_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
sentiment_to_idx = {'positive': 0, 'neutral': 1, 'negative': 2}

correct = 0
total = len(test_examples)

for i, example in enumerate(test_examples, 1):
    pred, probs = predict_sentiment(model, example["text"], enc)
    expected_idx = sentiment_to_idx[example["expected"]]
    
    is_correct = pred == expected_idx
    if is_correct:
        correct += 1
    
    print(f"\nExample {i}:")
    print("-" * 50)
    print(f"Conversation snippet:")
    print(example["text"].strip())
    print("\nPrediction:")
    print(f"Predicted: {sentiment_map[pred]}, Expected: {example['expected']}")
    print(f"Correct: {'✓' if is_correct else '✗'}")
    print(f"Confidence scores:")
    print(f"  Positive: {probs[0]:.3f}")
    print(f"  Neutral:  {probs[1]:.3f}")
    print(f"  Negative: {probs[2]:.3f}")
    print("-" * 50)

print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")

# Interactive mode
print("\nEnter your own conversations (type 'quit' to exit):")
while True:
    print("\nEnter conversation text (or 'quit' to exit):")
    lines = []
    while True:
        line = input()
        if line.lower() == 'quit':
            if not lines:  # If this is the first line
                exit()
            break
        lines.append(line)
        if not line:  # Empty line to finish input
            break
    
    conversation = '\n'.join(lines)
    if not conversation.strip():
        continue
        
    pred, probs = predict_sentiment(model, conversation, enc)
    print("\nPrediction:")
    print(f"Predicted sentiment: {sentiment_map[pred]}")
    print(f"Confidence scores:")
    print(f"  Positive: {probs[0]:.3f}")
    print(f"  Neutral:  {probs[1]:.3f}")
    print(f"  Negative: {probs[2]:.3f}")