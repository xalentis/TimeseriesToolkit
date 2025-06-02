import math
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class AutoLSTM:
    """
    Automated LSTM architecture that optimizes hyperparameters based on dataset characteristics.
    """
    
    def __init__(self, 
                 task_type: str = 'regression',
                 validation_size: float = 0.1,
                 random_state: int = 42,
                 device: str = None,
                 max_epochs: int = 100,
                 patience: int = 10,
                 memory_constraints: str = 'medium',
                 verbose: bool = False,
                 use_dropout: bool = False):
        """
        Initialize AutoLSTM with configuration parameters.
        
        Parameters:
        -----------
        task_type : str, default='regression'
            Type of task: 'regression', 'classification', or 'sequence_prediction'
        validation_size : float, default=0.1
            Proportion of training data to use for validation
        random_state : int, default=42
            Random seed for reproducibility
        device : str, optional
            Device to use ('cuda', 'mps', or 'cpu'). If None, will detect automatically.
        max_epochs : int, default=100
            Maximum number of training epochs
        patience : int, default=10
            Early stopping patience
        memory_constraints : str, default='medium'
            Level of memory constraints: 'low', 'medium', 'high'
        verbose : bool, default=False
            Whether to print detailed information
        use_dropout : bool, default=False
            Whether to apply dropout
        """
        self.task_type = task_type
        self.validation_size = validation_size
        self.random_state = random_state
        self.device = device or self._get_optimal_device()
        self.max_epochs = max_epochs
        self.patience = patience
        self.memory_constraints = memory_constraints
        self.verbose = verbose
        self.use_dropout = use_dropout
        
        # Model components (initialized after fit)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.architecture_params = None
        self.data_info = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Fit the AutoLSTM model to the data.
        
        Parameters:
        -----------
        df : pandas DataFrame
            The input DataFrame
        target_column : str, optional
            Target column for supervised learning
            
        Returns:
        --------
        self : AutoLSTM
            Returns self for method chaining
        """
        # Analyze dataset
        dataset_stats = self._analyze_dataset(df, target_column)
        
        if self.verbose:
            print(f"Dataset analysis: {dataset_stats['row_count']} rows, {dataset_stats['feature_count']} features")
        
        # Determine architecture parameters
        self.architecture_params = self._determine_architecture_params(
            dataset_stats, self.task_type, self.memory_constraints, self.use_dropout
        )
        
        if self.verbose:
            print(f"Architecture parameters: {self.architecture_params}")
        
        # Prepare data
        self.train_loader, self.val_loader, self.test_loader, self.data_info = self._prepare_sequences(
            df, target_column, self.architecture_params['sequence_length'],
            self.architecture_params['batch_size'], self.validation_size,
            self.random_state, self.task_type
        )
        
        # Determine model dimensions
        input_size = dataset_stats['feature_count']
        if target_column is not None:
            input_size -= 1
            
        if self.task_type == 'classification' and target_column is not None:
            output_size = len(df[target_column].unique())
        else:
            output_size = 1
        
        # Create model
        self.model = OptimalLSTM(
            input_size=input_size,
            hidden_size=self.architecture_params['hidden_size'],
            num_layers=self.architecture_params['num_layers'],
            output_size=output_size,
            dropout=self.architecture_params['dropout'],
            bidirectional=self.architecture_params['bidirectional'],
            task_type=self.task_type,
        ).to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = self._setup_optimizer()
        self.loss_fn = self._setup_loss_function()
        
        if self.verbose:
            print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        return self
    
    def train(self, epochs: Optional[int] = None):
        """
        Train the model.
        
        Parameters:
        -----------
        epochs : int, optional
            Number of epochs to train. If None, uses max_epochs from initialization.
            
        Returns:
        --------
        training_history : dict
            Dictionary containing training history
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        epochs = epochs or self.max_epochs
        return self._train_model(epochs)
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        data : DataFrame, ndarray, or Tensor
            Input data for prediction
            
        Returns:
        --------
        predictions : ndarray
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self._predict(data)
    
    def evaluate(self, test_data: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Parameters:
        -----------
        test_data : DataLoader, optional
            Test data loader. If None, uses the test set from fit.
            
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        test_loader = test_data or self.test_loader
        self.model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                
                predictions.extend(y_pred.cpu().numpy())
                targets.extend(y_batch.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        if self.task_type == 'regression':
            return {
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions)
            }
        else:
            # For classification, you'd implement accuracy, precision, recall, etc.
            return {'loss': mean_squared_error(targets, predictions)}
    
    def _get_optimal_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _analyze_dataset(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        row_count = len(df)
        feature_count = df.shape[1]
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_feature_count = numeric_df.shape[1]
        feature_variance = numeric_df.var().mean()
        
        # Calculate autocorrelation
        feature_autocorrelation = 0
        try:
            for col in numeric_df.columns:
                if len(numeric_df[col].dropna()) > 10:
                    autocorr = numeric_df[col].autocorr(lag=1)
                    if not np.isnan(autocorr):
                        feature_autocorrelation += abs(autocorr)
            if numeric_feature_count > 0:
                feature_autocorrelation /= numeric_feature_count
        except:
            feature_autocorrelation = 0.5
        
        return {
            'row_count': row_count,
            'feature_count': feature_count,
            'numeric_feature_count': numeric_feature_count,
            'feature_variance': feature_variance,
            'feature_autocorrelation': feature_autocorrelation
        }
    
    def _determine_architecture_params(self, dataset_stats: Dict[str, Any], 
                                     task_type: str, memory_constraints: str, 
                                     use_dropout: bool) -> Dict[str, Any]:
        """Determine optimal architecture parameters."""
        row_count = dataset_stats['row_count']
        feature_count = dataset_stats['feature_count']
        
        memory_scaling = {'low': 0.5, 'medium': 1.0, 'high': 2.0}.get(memory_constraints, 1.0)
        
        # Sequence length
        autocorr_factor = max(0.1, min(1.0, dataset_stats.get('feature_autocorrelation', 0.5)))
        if row_count < 1000:
            base_seq_length = max(3, min(10, int(row_count * 0.05)))
        elif row_count < 10000:
            base_seq_length = max(5, min(20, int(row_count * 0.02)))
        else:
            base_seq_length = max(10, min(50, int(row_count * 0.01)))
        
        sequence_length = max(3, int(base_seq_length * (0.5 + autocorr_factor)))
        
        # Batch size
        if row_count < 1000:
            batch_size = max(8, min(32, int(row_count * 0.1 * memory_scaling)))
        elif row_count < 10000:
            batch_size = max(16, min(64, int(row_count * 0.05 * memory_scaling)))
        else:
            batch_size = max(32, min(256, int(row_count * 0.02 * memory_scaling)))
        
        # Hidden size
        feature_factor = math.sqrt(feature_count)
        hidden_size = max(32, min(512, int(64 * feature_factor * memory_scaling)))
        hidden_size = 2 ** int(math.log2(hidden_size) + 0.5)  # Round to power of 2
        
        # Number of layers
        complexity_factor = max(0.5, min(1.5, dataset_stats.get('feature_variance', 1.0)))
        if row_count < 1000 or feature_count < 5:
            num_layers = 1
        elif row_count < 10000 or feature_count < 20:
            num_layers = max(1, min(2, int(1 * complexity_factor * memory_scaling)))
        else:
            num_layers = max(1, min(3, int(2 * complexity_factor * memory_scaling)))
        
        # Dropout
        if use_dropout and num_layers > 1:
            if row_count < 1000:
                dropout = 0.1
            elif row_count < 10000:
                dropout = 0.2
            else:
                dropout = 0.3
        else:
            dropout = 0.0
        
        # Bidirectional
        bidirectional = (task_type in ['classification', 'sequence_prediction'] and 
                        row_count >= 1000 and feature_count >= 5 and
                        memory_constraints != 'low')
        
        # Learning rate and weight decay
        if row_count < 1000:
            learning_rate = 0.01
            weight_decay = 1e-3
        elif row_count < 10000:
            learning_rate = 0.005
            weight_decay = 5e-4
        else:
            learning_rate = 0.001
            weight_decay = 1e-4
        
        # Optimizer
        optimizer_name = 'adamw' if (row_count >= 5000 and feature_count >= 10) else 'adam'
        
        return {
            'sequence_length': sequence_length,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer': optimizer_name
        }
    
    def _prepare_sequences(self, df, target_column, sequence_length, batch_size, 
                          validation_size, random_state, task_type):
        """Prepare LSTM sequences from data."""
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        shuffle = task_type != 'sequence_prediction'
        
        if target_column is not None and target_column in df.columns:
            X = df.drop(columns=[target_column]).values
            y = df[target_column].values
            
            X_sequences = []
            y_values = []
            for i in range(len(X) - sequence_length):
                X_sequences.append(X[i:i + sequence_length])
                y_values.append(y[i + sequence_length])
            
            X_sequences = np.array(X_sequences)
            y_values = np.array(y_values)
            X_tensor = torch.FloatTensor(X_sequences)
            y_tensor = torch.FloatTensor(y_values.reshape(-1, 1))
        else:
            data = df.values
            sequences = []
            for i in range(len(data) - sequence_length):
                sequences.append(data[i:i + sequence_length])
            
            sequences = np.array(sequences)
            data_tensor = torch.FloatTensor(sequences)
            X_tensor = data_tensor
            y_tensor = data_tensor
        
        # Split data
        indices = np.arange(len(X_tensor))
        if shuffle:
            np.random.shuffle(indices)
        
        test_size = int(0.1 * len(indices))
        val_size = int(validation_size * len(indices))
        train_size = len(indices) - test_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = torch.utils.data.TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        val_dataset = torch.utils.data.TensorDataset(X_tensor[val_indices], y_tensor[val_indices])
        test_dataset = torch.utils.data.TensorDataset(X_tensor[test_indices], y_tensor[test_indices])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        data_info = {
            'feature_dim': df.shape[1] - (1 if target_column else 0),
            'sequence_length': sequence_length,
            'has_target': target_column is not None,
            'target_column': target_column
        }
        
        return train_loader, val_loader, test_loader, data_info
    
    def _setup_optimizer(self):
        """Setup optimizer based on architecture parameters."""
        lr = self.architecture_params['learning_rate']
        weight_decay = self.architecture_params['weight_decay']
        optimizer_name = self.architecture_params.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        return optimizer
    
    def _setup_loss_function(self):
        """Setup loss function based on task type."""
        if self.task_type == 'classification':
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()
    
    def _train_model(self, max_epochs):
        """Train the model with early stopping."""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(max_epochs):
            self.model.train()
            epoch_train_loss = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch, y_batch, epoch)
                
                if y_pred.shape != y_batch.shape:
                    if len(y_pred.shape) > len(y_batch.shape):
                        y_batch = y_batch.long().view(-1)
                    else:
                        y_pred = y_pred.view(y_batch.shape)
                
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    
                    if y_pred.shape != y_batch.shape:
                        if len(y_pred.shape) > len(y_batch.shape):
                            y_batch = y_batch.long().view(-1)
                        else:
                            y_pred = y_pred.view(y_batch.shape)
                    
                    loss = self.loss_fn(y_pred, y_batch)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)
            
            if self.scheduler:
                self.scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }
    
    def _predict(self, data):
        """Make predictions on new data."""
        self.model.eval()
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if isinstance(data, np.ndarray):
            if len(data.shape) == 2:
                sequences = []
                seq_len = self.architecture_params['sequence_length']
                for i in range(0, len(data) - seq_len + 1):
                    sequences.append(data[i:i + seq_len])
                if len(sequences) == 0:
                    raise ValueError(f"Input data length too short for sequence length {seq_len}")
                data = np.array(sequences)
            data = torch.FloatTensor(data)
        
        data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                else:
                    inputs = batch[0].to(self.device)
                    outputs = self.model(inputs)
                
                if self.task_type == 'classification':
                    if outputs.shape[-1] > 1:
                        batch_preds = torch.softmax(outputs, dim=-1).cpu().numpy()
                    else:
                        batch_preds = torch.sigmoid(outputs).cpu().numpy()
                else:
                    batch_preds = outputs.cpu().numpy()
                predictions.append(batch_preds)
        
        return np.vstack(predictions)


class OptimalLSTM(nn.Module):
    """Optimized LSTM model with attention mechanism."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, 
                 dropout=0.0, bidirectional=False, task_type='regression'):
        super(OptimalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.task_type = task_type
        self.input_size = input_size
        self.direction_factor = 2 if bidirectional else 1
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.direction_factor, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # FC layers
        fc_layers = []
        fc_layers.append(nn.Linear(hidden_size * self.direction_factor, hidden_size))
        fc_layers.append(nn.LayerNorm(hidden_size))
        fc_layers.append(nn.ReLU())
        if dropout > 0.0:
            fc_layers.append(nn.Dropout(dropout))
        
        second_fc_size = hidden_size // 2
        fc_layers.append(nn.Linear(hidden_size, second_fc_size))
        fc_layers.append(nn.LayerNorm(second_fc_size))
        fc_layers.append(nn.ReLU())
        if dropout > 0.0:
            fc_layers.append(nn.Dropout(dropout/2))
        
        fc_layers.append(nn.Linear(second_fc_size, output_size))
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def _apply_attention(self, lstm_output):
        attn_weights = self.attention(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights, dim=1)
        return context
    
    def forward(self, x, target=None, epoch=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Batch normalization
        x_reshaped = x.reshape(-1, x.size(-1))
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, -1)
        
        # LSTM forward pass
        h0 = torch.zeros(self.num_layers * self.direction_factor, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.direction_factor, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        context = self._apply_attention(lstm_out)
        
        # Final prediction
        out = self.fc_layers(context)
        return out
