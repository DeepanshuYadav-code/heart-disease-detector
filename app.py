from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('heart_disease_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the formgit
        features = [float(request.form[f]) for f in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
        
        # Scale the input features
        scaled_features = scaler.transform(np.array(features).reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        result = "You may have heart disease. Please consult a doctor." if prediction == 1 else "You are likely healthy, but always consult a doctor for proper medical advice."
        
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
