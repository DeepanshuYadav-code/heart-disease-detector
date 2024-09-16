import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def train_model():
    # Load the data
    data = pd.read_csv('heart.csv')

    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the model and scaler
    joblib.dump(model, 'heart_disease_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

if __name__ == '__main__':
    train_model()
