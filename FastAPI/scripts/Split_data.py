import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(input_path):
    df = pd.read_csv(input_path)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    y = df['Genre'].map({'Male': 0, 'Female': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
