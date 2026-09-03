import pandas as pd
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from train import train_model, evaluate_model

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
MODEL_PATH = os.path.join(MODEL_DIR, "07_01_feed_forward.pt")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "07_01_feed_forward_checkpoint.pt")
TENSORBOARD_DIR = os.path.join(BASE_DIR, "../../runs/feed_forward")

TRAIN_FILE = os.path.join(SPLIT_DATA_DIR, "train.csv")
VAL_FILE   = os.path.join(SPLIT_DATA_DIR, "val.csv")
TEST_FILE  = os.path.join(SPLIT_DATA_DIR, "test.csv")

# Load parquet files
X_train = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "X_train.parquet")).values
y_train = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "y_train.parquet")).values

X_val   = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "X_val.parquet")).values
y_val   = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "y_val.parquet")).values

X_test  = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "X_test.parquet")).values
y_test  = pd.read_parquet(os.path.join(POST_SPLIT_DIR, "y_test.parquet")).values

# check log
print("X_train min/max:", X_train.min(), X_train.max())
print("y_train min/max:", y_train.min(), y_train.max())

# ---------------------
# Define the MLP model (IMPROVED for multi-horizon)
# ---------------------
class CryptoMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Deeper network with layer norm, batch norm and dropout for multi-task learning
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Normalize input features
        x = self.input_norm(x)
        return self.net(x)

batch_size = 64

# Convert numpy → tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
test_loader  = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                        torch.tensor(y_test, dtype=torch.float32)), 
                          batch_size=batch_size)


# training loop
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Training on device:", device)

print(X_train.shape, y_train.shape)
model = CryptoMLP(input_dim=X_train.shape[1], output_dim=y_train.shape[1]).to(device)
criterion = nn.MSELoss()  # for regression; use BCEWithLogitsLoss() if classification
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Lower LR + weight decay

# Add learning rate scheduler with more patience for multi-task learning
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5  # Increased patience from 3 to 5
)

# Optional: Define custom metrics function
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
print("Starting Training")
print("="*50)

training_results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=100,
    model_save_path=MODEL_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    scheduler=scheduler,
    early_stopping_patience=10,  # Increased patience for multi-task learning
    early_stopping_min_delta=1e-5,  # More sensitive to small improvements
    tensorboard_log_dir=TENSORBOARD_DIR,
    metric_fn=compute_metrics,
    gradient_clip_val=1.0,
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

# Get predictions for further analysis
model.eval()
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())

y_pred = np.concatenate(all_predictions, axis=0)
y_true = np.concatenate(all_targets, axis=0)

