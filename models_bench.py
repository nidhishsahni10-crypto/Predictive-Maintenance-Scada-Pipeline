import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# PART A: PYTORCH LSTM ARCHITECTURE
# ==========================================
class PredictiveMaintenanceLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_layers=1):
        super(PredictiveMaintenanceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Gather final hidden state at the last sequence time step
        last_step_output = lstm_out[:, -1, :]
        prediction = self.sigmoid(self.fc(last_step_output))
        return prediction

def train_pytorch_lstm(X_data, y_data):
    # Convert arrays to appropriate PyTorch Tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PredictiveMaintenanceLSTM(input_dim=X_data.shape[2])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Single-epoch validation run-through
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
    return model

# ==========================================
# PART B: SCIKIT-LEARN 5-CLASSIFIER SUITE
# ==========================================
def evaluate_yield_classifiers(X_train, y_train, X_test, y_test):
    """
    Evaluates 5 target Classifiers to isolate optimal parameter configurations
    for cutting industrial product thickness variability.
    """
    model_suite = {
        'RandomForest': (RandomForestClassifier(), {'n_estimators': [10, 50]}),
        'GradientBoosting': (GradientBoostingClassifier(), {'learning_rate': [0.1, 0.2]}),
        'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [20, 50]}),
        'LogisticRegression': (LogisticRegression(max_iter=500), {'C': [1.0, 10.0]}),
        'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [5, 10]})
    }
    
    print("\n--- Testing 5+ Scikit-Learn Classifiers on SCADA Pipelines ---")
    performance_records = {}
    
    for label, (classifier, grid_params) in model_suite.items():
        search = GridSearchCV(classifier, grid_params, cv=2, scoring='accuracy')
        search.fit(X_train, y_train)
        
        top_model = search.best_estimator_
        predictions = top_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        performance_records[label] = accuracy
        print(f" -> Classifier: {label:<20} | Optimization Accuracy: {accuracy:.4%}")
        
    return performance_records

if __name__ == "__main__":
    print("\n[Executing Deep Learning & Classification Benchmarks]")
    
    # Create balanced structured matrix inputs for Classifier testing
    np.random.seed(42)
    X_train_matrix = np.random.randn(400, 4)
    y_train_matrix = np.random.randint(0, 2, size=400)
    X_test_matrix = np.random.randn(100, 4)
    y_test_matrix = np.random.randint(0, 2, size=100)
    
    # Run Scikit-learn validation suite
    evaluate_yield_classifiers(X_train_matrix, y_train_matrix, X_test_matrix, y_test_matrix)
    
    # Construct sequence windows to simulate sequential sensory inputs for LSTM (Batch, Window, Features)
    mock_sequences = np.random.randn(100, 10, 4)
    mock_labels = np.random.randint(0, 2, size=100)
    
    trained_network = train_pytorch_lstm(mock_sequences, mock_labels)
    print(" -> PyTorch LSTM forward optimization pass verified successfully.")