import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Optional, Callable, Dict, Any
import time


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    model_save_path: str,
    checkpoint_path: Optional[str] = None,
    scheduler: Optional[Any] = None,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    tensorboard_log_dir: Optional[str] = None,
    metric_fn: Optional[Callable] = None,
    gradient_clip_val: Optional[float] = None,
    save_best_only: bool = True,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Generic training function for PyTorch models with TensorBoard logging and early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Maximum number of epochs to train
        model_save_path: Path to save the best model
        checkpoint_path: Path to load/save checkpoints (optional)
        scheduler: Learning rate scheduler (optional)
        early_stopping_patience: Number of epochs to wait for improvement
        early_stopping_min_delta: Minimum change to qualify as improvement
        tensorboard_log_dir: Directory for TensorBoard logs (optional)
        metric_fn: Additional metric function that takes (y_pred, y_true) and returns dict of metrics
        gradient_clip_val: Value for gradient clipping (optional)
        save_best_only: If True, only save model when validation loss improves
        verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
    
    Returns:
        Dictionary containing training history and final metrics
    """
    
    # Initialize TensorBoard writer
    writer = None
    if tensorboard_log_dir:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
    
    # Load checkpoint if it exists
    start_epoch = 0
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', history)
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if verbose > 0:
            print(f"Resuming training from epoch {start_epoch}")
    
    # Early stopping variables
    epochs_no_improve = 0
    training_start_time = time.time()
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # ==================== Training Phase ====================
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Log batch-level metrics to TensorBoard
            if writer and batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        
        avg_train_loss = train_loss / train_batches
        
        # ==================== Validation Phase ====================
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_outputs = []
        all_val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Store for metric calculation
                if metric_fn:
                    all_val_outputs.append(outputs.cpu())
                    all_val_targets.append(targets.cpu())
        
        avg_val_loss = val_loss / val_batches
        
        # Calculate additional metrics
        additional_metrics = {}
        if metric_fn and all_val_outputs:
            all_val_outputs = torch.cat(all_val_outputs, dim=0)
            all_val_targets = torch.cat(all_val_targets, dim=0)
            additional_metrics = metric_fn(all_val_outputs, all_val_targets)
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(current_lr)
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log additional metrics
            for metric_name, metric_value in additional_metrics.items():
                writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # Log model gradients and weights
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                writer.add_histogram(f'Weights/{name}', param, epoch)
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        if verbose >= 2:
            metrics_str = ' '.join([f'{k}={v:.6f}' for k, v in additional_metrics.items()])
            print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.2f}s] - "
                  f"train_loss: {avg_train_loss:.6f} - val_loss: {avg_val_loss:.6f} - "
                  f"lr: {current_lr:.6f}" + (f" - {metrics_str}" if metrics_str else ""))
        
        # ==================== Model Checkpointing ====================
        improved = avg_val_loss < (best_val_loss - early_stopping_min_delta)
        
        if improved:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            # Save best model
            if save_best_only or model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss,
                    'history': history
                }, model_save_path)
                
                if verbose > 0:
                    print(f"  → Saved best model (val_loss: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
        
        # Save checkpoint (for resuming training)
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'history': history
            }, checkpoint_path)
        
        # ==================== Early Stopping ====================
        if epochs_no_improve >= early_stopping_patience:
            if verbose > 0:
                print(f"\nEarly stopping triggered after {epoch+1} epochs "
                      f"(no improvement for {early_stopping_patience} epochs)")
            break
    
    # Training complete
    total_time = time.time() - training_start_time
    
    if verbose > 0:
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'total_epochs': len(history['train_loss']),
        'total_time': total_time,
        'history': history
    }


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> nn.Module:
    """
    Load a model from a checkpoint.
    
    Args:
        model: Model instance with the same architecture
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
    
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    metric_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
        metric_fn: Optional function to calculate additional metrics
    
    Returns:
        Dictionary containing test loss and metrics
    """
    model.eval()
    test_loss = 0.0
    test_batches = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            test_batches += 1
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    avg_test_loss = test_loss / test_batches
    results = {'test_loss': avg_test_loss}
    
    # Calculate additional metrics
    if metric_fn and all_outputs:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        additional_metrics = metric_fn(all_outputs, all_targets)
        results.update(additional_metrics)
    
    return results
