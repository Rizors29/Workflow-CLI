import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Enable autolog
mlflow.autolog()

# Load dataset preprocessing
X_train = pd.read_csv('titanic_preprocessing/X_train.csv')
X_test = pd.read_csv('titanic_preprocessing/X_test.csv')

y_train = pd.read_csv('titanic_preprocessing/y_train.csv')
y_test = pd.read_csv('titanic_preprocessing/y_test.csv')

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

with mlflow.start_run():
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")

    # Log metrics
    print(classification_report(y_test, y_pred))
    mlflow.log_metric("accuracy", accuracy)