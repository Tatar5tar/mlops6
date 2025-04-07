from Clean_data import clean_data
from Split_data import split_data
from Train_model import train_model
from Eval_model import eval_model

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
import pandas as pd


RAW_DATA_PATH = "data/mall_customers.csv"
CLEAN_DATA_PATH = "data/clean_mall_customers.csv"
MODEL_PATH = "model/model.pkl"
METRICS_PATH = "metrics/metrics.json"

app = FastAPI(title='lab6')
model = None


def load_model(path):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print('Модель загружена')

    except Exception as e:
        print(f'Не удалось загрузить модель: {e}')
        model = None

    return model


class CustomerFeatures(BaseModel):
    customer_id: int
    genre: str
    age: int
    annual_income: int
    spending_score: int


@app.post('/predict', summary='Predict customer\'s gender')
async def predict(customer: CustomerFeatures):
    try:
        if model is None:
            return {'error': 'Модель не загружена'}

        input_df = pd.DataFrame([customer.model_dump()])
        prediction = model.predict(input_df)[0]

        return {'prediction': int(prediction)}

    except Exception as e:
        return {'error': f'Ошибка: {e}'}


def pipe():
    clean_data(RAW_DATA_PATH, CLEAN_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(CLEAN_DATA_PATH)

    model_path = train_model(X_train, y_train, MODEL_PATH)
    eval_model(model_path, X_test, y_test, METRICS_PATH)


if __name__ == '__main__':
    pipe()
    model = load_model(MODEL_PATH)
    uvicorn.run(app, host='0.0.0.0', port=8005)

