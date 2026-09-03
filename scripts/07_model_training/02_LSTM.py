import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from train import train_model, evaluate_model
import gc

#config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
LARGE_DATA_DIR = os.path.join(BASE_DIR, "../../data/large_data")
SPLIT_DATA_DIR = os.path.join(BASE_DIR, "../../data/split_data")
IMAGE_DIR = os.path.join(BASE_DIR, "../../images/03_features_plotting")
IMAGE_DIR_ORG = os.path.join(BASE_DIR, "../../images")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR_ORG, exist_ok=True)
DATASET_FILE = os.path.join(LARGE_DATA_DIR, "03_crypto_ml_dataset.csv")
POST_SPLIT_DIR = os.path.join(BASE_DIR, "../../data/post_split_data")
os.makedirs(POST_SPLIT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(BASE_DIR, "../../models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "07_02_LSTM.pt")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "07_02_LSTM_checkpoint.pt")
TENSORBOARD_DIR = os.path.join(BASE_DIR, "../../runs/lstm")

TRAIN_FILE = os.path.join(SPLIT_DATA_DIR, "train.csv")
VAL_FILE   = os.path.join(SPLIT_DATA_DIR, "val.csv")
TEST_FILE  = os.path.join(SPLIT_DATA_DIR, "test.csv")

TRAIN_SCALED_FILE = os.path.join(SPLIT_DATA_DIR, "train_scaled.csv")
VAL_SCALED_FILE   = os.path.join(SPLIT_DATA_DIR, "val_scaled.csv")
TEST_SCALED_FILE  = os.path.join(SPLIT_DATA_DIR, "test_scaled.csv")
SCALER_PATH = os.path.join(LARGE_DATA_DIR, "05_feature_scaler.pkl")

# Load parquet files
print("Loading data...")
X_train = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "X_train.parquet")).values
y_train = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "y_train.parquet")).values

X_val   = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "X_val.parquet")).values
y_val   = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "y_val.parquet")).values

X_test  = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "X_test.parquet")).values
y_test  = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "y_test.parquet")).values

print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")


# Memory-efficient sequence dataset
class SequenceDataset(Dataset):
    """
    Memory-efficient dataset that creates sequences on-the-fly
    instead of pre-allocating all sequences in memory.
    
    IMPORTANT: For val/test sets, this should only be used with data that 
    doesn't overlap with the previous split to prevent leakage.
    """
    def __init__(self, X, y, seq_len, start_idx=0):
        """
        Args:
            X: Feature array
            y: Target array
            seq_len: Sequence length
            start_idx: Starting index for sequences (used to skip overlapping data)
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.start_idx = start_idx
        
    def __len__(self):
        return len(self.X) - self.seq_len - self.start_idx
    
    def __getitem__(self, idx):
        # Adjust index by start_idx
        actual_idx = idx + self.start_idx
        X_seq = self.X[actual_idx:actual_idx + self.seq_len]
        y_seq = self.y[actual_idx + self.seq_len]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


# Create sequence datasets (memory-efficient)
seq_len = 16  # Match the 4-hour prediction horizon (16 x 15min = 4h)
batch_size = 64

print(f"Creating sequence datasets with seq_len={seq_len}...")
print(f"NOTE: Val/test datasets start at index {seq_len} to prevent train/val/test overlap leakage")

# Training dataset starts from index 0
train_dataset = SequenceDataset(X_train, y_train, seq_len, start_idx=0)

# CRITICAL: Val/test datasets skip first seq_len rows to prevent using previous split data
# This ensures sequences only use data from their own split
val_dataset = SequenceDataset(X_val, y_val, seq_len, start_idx=seq_len)
test_dataset = SequenceDataset(X_test, y_test, seq_len, start_idx=seq_len)

print(f"Sequence dataset sizes (after preventing overlap):")
print(f"  Train: {len(train_dataset)} sequences")
print(f"  Val:   {len(val_dataset)} sequences (skipped first {seq_len} rows)")
print(f"  Test:  {len(test_dataset)} sequences (skipped first {seq_len} rows)")

# Use pin_memory=True for faster GPU transfer, num_workers for parallel loading
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                        num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         num_workers=2, pin_memory=True)

# Clean up to free memory
del X_train, X_val, X_test, y_train, y_val, y_test
gc.collect()

# define LSTM model (MEMORY-EFFICIENT VERSION)
class CryptoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layer normalization for input features
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Layer normalization after LSTM
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Normalize input features
        x = self.input_norm(x)
        
        # LSTM initializes hidden states to zero by default if not provided
        # This is more memory-efficient than creating them manually

        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]  # take last timestep
        out = self.output_norm(out)  # normalize before final layer
        out = self.fc(out)
        return out
    
# train LSTM model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")

# Get input/output dimensions from dataset
sample_x, sample_y = train_dataset[0]
input_dim = sample_x.shape[1]  # num_features
output_dim = sample_y.shape[0]  # num_targets

print(f"Model dimensions: input_dim={input_dim}, output_dim={output_dim}")

model = CryptoLSTM(
    input_dim=input_dim,
    hidden_dim=64,  # Reduced from 128 to save memory
    output_dim=output_dim,
    num_layers=2,
    dropout=0.2
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Custom metrics for LSTM
def compute_metrics(y_pred, y_true):
    """Compute additional metrics for regression"""
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    return {
        'mae': mae,
        'rmse': rmse
    }

# Train the model using the generic training function
print("\n" + "="*50)
print("Starting LSTM Training")
print("="*50)

training_results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=50,  # Reduced from 100
    model_save_path=MODEL_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    scheduler=scheduler,
    early_stopping_patience=7,
    early_stopping_min_delta=1e-5,
    tensorboard_log_dir=TENSORBOARD_DIR,
    metric_fn=compute_metrics,
    gradient_clip_val=1.0,  # Critical for LSTM to prevent exploding gradients
    save_best_only=True,
    verbose=2
)

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"Best validation loss: {training_results['best_val_loss']:.6f}")
print(f"Total epochs: {training_results['total_epochs']}")
print(f"Total time: {training_results['total_time']:.2f}s")

# Evaluate on test set
print("\n" + "="*50)
print("Evaluating on Test Set")
print("="*50)

# Load best model for evaluation
from train import load_model_checkpoint
model = load_model_checkpoint(model, MODEL_PATH, device)

test_results = evaluate_model(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device=device,
    metric_fn=compute_metrics
)

print(f"Test Loss: {test_results['test_loss']:.6f}")
print(f"Test MAE: {test_results['mae']:.6f}")
print(f"Test RMSE: {test_results['rmse']:.6f}")

print("\n✅ LSTM training completed successfully!")
print(f"Model saved to: {MODEL_PATH}")
print(f"TensorBoard logs: {TENSORBOARD_DIR}")
print("\nTo view training logs, run:")
print(f"  tensorboard --logdir=runs")
