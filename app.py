from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("heart_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input
        input_data = {
            'age': int(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': int(request.form['trestbps']),
            'chol': int(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': int(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Match training preprocessing
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        input_df[categorical_cols] = input_df[categorical_cols].astype('category')

        # Predict
        processed = preprocessor.transform(input_df)
        probability = model.predict_proba(processed)[0][1] * 100

        result = "High Chance of Heart Disease ❌" if probability > 50 else "Low Chance of Heart Disease ✅"

        return render_template('index.html', prediction=result, probability=f"{probability:.2f}%")
    
    except Exception as e:
        return render_template('index.html', prediction="⚠️ Error processing input", probability=str(e))


if __name__ == '__main__':
    app.run(debug=True)
