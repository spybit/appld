from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    float_features = []
    for key, value in request.form.items():
        if key == 'Gender':
            # Convert 'Gender' to 0 for 'Female' and 1 for 'Male'
            numeric_value = 0 if value.lower() == 'Female' else 1
        else:
            # Convert other features to float
            try:
                numeric_value = float(value)
            except ValueError:
                # Handle non-numeric values (if needed)
                numeric_value = None  # You can set a default value or handle it in a different way
        float_features.append(numeric_value)
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

