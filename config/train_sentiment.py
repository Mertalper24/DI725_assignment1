# train the sentiment analysis model
out_dir = 'out-sentiment'
eval_interval = 50
eval_iters = 20
log_interval = 10

# Save best model
always_save_checkpoint = True

wandb_log = False
wandb_project = 'sentiment-analysis2'
wandb_run_name = 'gpt-sentiment-scratch2'

dataset = 'customer_service'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512

# Model settings
n_layer = 4
n_head = 8
n_embd = 256
dropout = 0.2  # Moderate dropout
num_classes = 3

# Training settings
learning_rate = 2e-5  # Moderate learning rate
max_iters = 1000
weight_decay = 0.1  # Moderate weight decay
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# Learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 800
min_lr = 2e-6

# Class balancing - based on inverse of class frequency
# For class distribution: Positive: 1.75%, Neutral: 55.88%, Negative: 42.37%
class_weights = [32.0, 1.0, 1.3]  # Much higher weight for the rare positive class