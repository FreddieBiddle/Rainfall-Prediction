import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm import tqdm
from xgboost import XGBClassifier
import feature_engineering as fe

train = fe.train
test = fe.test

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"ROC AUC Score: {roc_auc:.5f}\n")

predictors = ['day', 'pressure', 'maxtemp', 'temperature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed', 'rainfall', 'year_group', 'temperature_range',
       'seasonal_sin', 'day_maxtemp', 'day_sunshine', 'day_winddirection',
       'day_year_group', 'pressure_maxtemp', 'pressure_mintemp',
       'pressure_temperature_range', 'maxtemp_dewpoint', 'maxtemp_year_group',
       'maxtemp_seasonal_sin', 'mintemp_winddirection',
       'dewpoint_winddirection', 'dewpoint_year_group', 'humidity_year_group',
       'cloud_windspeed', 'windspeed_year_group']
target = 'rainfall'

n = 4 * 365
df_train = train.iloc[:n]
df_test = train.iloc[n:]

df_train['hp'] = df_train.groupby('day')['rainfall'].mean().to_list() * 4
df_test['hp'] = df_train.groupby('day')['rainfall'].mean().to_list() * 2

X_train = df_train[predictors]
y_train = df_train[target]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = df_test[predictors]
y_test = df_test[target]

X_test = scaler.transform(X_test)

models = {
    "SVM": SVC(kernel='poly', degree=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, min_child_weight=3, subsample=0.8, \
                             colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss", random_state=42)
    }

param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

for model in models.values():
    train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

X = train[predictors].values
y = train[target].values

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM [samples, time steps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split into train and test sets
N = 4 * 365
X_train, X_test, y_train, y_test = X_reshaped[:N], X_reshaped[N:], y[:N], y[N:]
# X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1) # Reshape y to [samples, 1]
X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64).view(-1, 1)

# Create DataLoader for efficient batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take the last time step's output
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Instantiate the model
input_size = X_train_tensor.shape[2]  # Number of features
hidden_size = 50
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

        accuracy = correct / total
        roc_auc = roc_auc_score(all_labels, all_predictions) # Calculate ROC AUC score
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}')

# Final evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(outputs.cpu().numpy())

accuracy = correct / total
roc_auc = roc_auc_score(all_labels, all_predictions)  # Calculate ROC AUC score
print(f'Final Test Accuracy: {accuracy:.4f}, Final ROC AUC: {roc_auc:.4f}')
