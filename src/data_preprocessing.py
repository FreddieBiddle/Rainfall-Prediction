mport pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_train_data(path):
    train = pd.read_csv(path)
    train = train.rename(columns={'temparature': 'temperature'})  # Fix typo
    return train

def load_test_data(path):
    test = pd.read_csv(path)
    test = test.rename(columns={'temparature': 'temperature'})  # Fix typo
    return test

def preprocess_data(df, predictors, target):
    X = df[predictors]
    y = df[target]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    return train_test_split(X, y, test_size=0.2, random_state=42)
