import os
import pickle
import json
from sklearn.metrics import accuracy_score, log_loss, f1_score


def eval_model(model_path, X_test, y_test, metrics_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    new_metrics = {
        'accuracy': accuracy,
        'log_loss': loss,
        'f1_score': f1
    }

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

    else:
        metrics_data = {'metrics': []}

    metrics_data['metrics'].append(new_metrics)

    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Accuracy = {accuracy}, Log_loss = {loss}, F1_score = {f1}")
