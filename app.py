from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle

# Load model and scalers
model = pickle.load(open('//Users//sadhvi//Downloads//Crop-prediction---Backend-main//model.pkl', 'rb'))
sc = pickle.load(open('//Users//sadhvi//Downloads//Crop-prediction---Backend-main//standscaler.pkl', 'rb'))
ms = pickle.load(open('/Users//sadhvi//Downloads//Crop-prediction---Backend-main//minmaxscaler.pkl', 'rb'))

app = Flask(__name__)
CORS(app)

# Crop labels mapped assuming 0-indexed classes from model
crop_dict = {
    0: "Rice", 1: "Maize", 2: "Jute", 3: "Cotton", 4: "Coconut", 5: "Papaya", 6: "Orange",
    7: "Apple", 8: "Muskmelon", 9: "Watermelon", 10: "Grapes", 11: "Mango", 12: "Banana",
    13: "Pomegranate", 14: "Lentil", 15: "Blackgram", 16: "Mungbean", 17: "Mothbeans",
    18: "Pigeonpeas", 19: "Kidneybeans", 20: "Chickpea", 21: "Coffee"
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict_html():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled = ms.transform(features)
        final = sc.transform(scaled)

        prediction = model.predict(final)[0]
        crop = crop_dict.get(prediction, "Unknown")
        result = f"{crop} is the best crop to be cultivated right there"
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

@app.route("/api/predict", methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data['Nitrogen']),
            float(data['Phosphorus']),
            float(data['Potassium']),
            float(data['Temperature']),
            float(data['Humidity']),
            float(data['Ph']),
            float(data['Rainfall']),
        ]])

        scaled = ms.transform(features)
        final = sc.transform(scaled)
        prediction = model.predict(final)[0]
        crop = crop_dict.get(prediction, "Unknown")

        return jsonify({'prediction': crop})
    except Exception as e:
        return jsonify({'prediction': 'Error', 'message': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
