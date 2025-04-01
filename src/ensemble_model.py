mport numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
%run feature_engineering.ipynb

# Preparing ensemble
X = train.loc[:, train.columns != 'rainfall']
y = train['rainfall']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
}
rf = RandomizedSearchCV(RandomForestClassifier(), rf_param_grid, n_iter=20, cv=5, n_jobs=-1)
rf.fit(X_train, y_train)
best_rf = rf.best_estimator_

# Hyperparameter tuning for XGBoost
xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    
}
xgb = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, n_iter=20, cv=5, n_jobs=-1)
xgb.fit(X_train, y_train)
best_xgb = xgb.best_estimator_

# Adding LightGBM
lgb = LGBMClassifier(n_estimators=200)
lgb.fit(X_train, y_train)

# ExtraTrees model
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)


# Ensemble Model

# ensemble = StackingClassifier(
#     estimators=[
#         ('rf', best_rf),
#         ('xgb', best_xgb),
#         ('lgb', lgb), 
#         ('et', et_model)
#     ],
#     final_estimator=LogisticRegression(),
#     cv=5  # Cross-validation for meta-model
# )


ensemble = VotingClassifier(estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lgb', lgb), 
        ('et', et_model)
], voting='soft', weights=[2, 3, 3, 3])  # Weighted based on model performance
    
    ensemble.fit(X_train, y_train)
    ensemble_acc = ensemble.score(X_test, y_test)
    print(f"Ensemble Model Accuracy: {ensemble_acc:.4f}")

    # Define Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
                                                                                            
    def forward(self, x):
    x = self.relu(self.batchnorm1(self.layer1(x)))
    x = self.dropout(x)
    x = self.relu(self.batchnorm2(self.layer2(x)))
    x = self.dropout(x)
    x = self.relu(self.layer3(x))
    x = self.output(x)
    return torch.sigmoid(x)

# Convert data for PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train Neural Network
input_size = X_train.shape[1]
model = NeuralNet(input_size)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
criterion = nn.BCELoss()

def train_nn(model, dataloader, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            X_batch, y_batch = batch
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss: .4f}")

train_nn(model, dataloader, optimizer, criterion)

# Evaluate Neural Network
y_pred_test = model(X_test_tensor).detach().numpy()
y_pred_test = (y_pred_test > 0.5).astype(int)
nn_acc = np.mean(y_pred_test == y_test_tensor.numpy())
print(f"Neural Network Accuracy: {nn_acc: .4f}")
