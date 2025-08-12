from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('crop_data.csv')
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = [[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall']),
        ]]
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)[0]
        crop = label_encoder.inverse_transform([prediction])[0]
        return jsonify({'crop': crop})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
