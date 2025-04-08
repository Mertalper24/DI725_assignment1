import time

# Output directory for fine-tuned model
out_dir = 'out-sentiment-ft'
eval_interval = 50
eval_iters = 20
log_interval = 10

# Save best model
always_save_checkpoint = True

# Wandb logging (optional)
wandb_log = False
wandb_project = 'sentiment-ft'
wandb_run_name = 'ft-' + str(time.time())

# Dataset config
dataset = 'customer_service'
init_from = 'gpt2'  # Initialize from pre-trained GPT-2

# Training settings for fine-tuning
batch_size = 32
gradient_accumulation_steps = 1
block_size = 512

# Model settings (keep same as original for compatibility)
n_layer = 12  # GPT-2 base has 12 layers
n_head = 12
n_embd = 768
dropout = 0.2
num_classes = 3

# Fine-tuning specific settings
learning_rate = 2e-5  # Lower learning rate for fine-tuning
max_iters = 1000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# Learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 800
min_lr = 2e-6

# Class weights based on your distribution
class_weights = [32.0, 1.0, 1.3]  # positive, neutral, negative 