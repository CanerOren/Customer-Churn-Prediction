from keras.models import load_model
from flask import Flask, request,jsonify,render_template
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')




def predict():
    model = load_model('models/ann.h5')
    path = 'data.json'

    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame([data])

    int_cols = ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Satisfaction Score',
                'Point Earned']
    float_cols = ['Balance', 'EstimatedSalary']
    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)

    X = pd.read_csv('X_nonprocessed.csv')

    merged = pd.concat([X, df], ignore_index=True)

    hot = pd.get_dummies(merged[["Geography", "Gender", "Card Type"]])
    merged = pd.concat([merged, hot], axis=1)
    merged = merged.drop(["Geography", "Gender", "Card Type"], axis=1)

    scaler = StandardScaler()
    merged = scaler.fit_transform(merged)

    processed_data = merged[-1, :]
    processed_data = processed_data.reshape(1, -1)
    prediction = model.predict(processed_data)
    predicted_label = (prediction > 0.5).astype(int)
    if(predicted_label==1):
        return f"Müşterimiz ayrılacak lütfen iletişime geçin."
    else:
        return f"Müşterimiz ayrılmayacak bir eyleme geçmeyin."

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form.to_dict()
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)
    text = predict()
    return render_template('index.html',prediction_text=text)

if __name__ == "__main__":
    app.run()

