from sklearn.ensemble import RandomForestClassifier
import pickle


def train_model(X_train, y_train, model_path):
    hyperparams = {'n_estimators': 50, 'max_depth': 7}

    model = RandomForestClassifier(n_estimators=hyperparams['n_estimators'], max_depth=hyperparams['max_depth'], random_state=42)
    model.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Модель обучена с гиперпараметрами: {hyperparams}")

    return model_path
