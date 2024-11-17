from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:3011"}})

# Load saved objects
weather_model = joblib.load('weather_model_r.pkl')  # Model for weather prediction
rain_model = joblib.load('weather_model_rain.pkl')  # Model for rain prediction
rain_amount_model = joblib.load('weather_model_rain_amount.pkl')  # Model for rain amount prediction
scaler = joblib.load('scaler.pkl')
scaler_rain = joblib.load('scaler_rain.pkl')
scaler_rain_amount = joblib.load('scaler_rain_amount.pkl')
encoder = joblib.load('encoder.pkl')
X_final = joblib.load('X_final.pkl')

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/test', methods=['POST'])
def test():
    data = request.json  # Get the JSON data sent to the endpoint
    return jsonify(data), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    province = data.get('province')
    date_str = data.get('date')
    
    if not province or not date_str:
        return jsonify({'error': 'Please provide both province and date.'}), 400

    try:
        date = pd.to_datetime(date_str)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

    # Prepare input data
    df_input = pd.DataFrame({
        'province': [province],
        'date': [date],
    })

    # Data preprocessing
    df_input['year'] = df_input['date'].dt.year
    df_input['month'] = df_input['date'].dt.month
    df_input['day'] = df_input['date'].dt.day
    df_input['day_of_week'] = df_input['date'].dt.day_name()
    df_input['week_of_year'] = df_input['date'].dt.isocalendar().week
    df_input['year_quarter'] = df_input['date'].dt.to_period('Q').astype('int')
    df_input['month_period'] = df_input['date'].dt.to_period('M').astype('int')

    day_of_week_mapping = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }

    df_input['day_of_week_encoded'] = df_input['day_of_week'].map(day_of_week_mapping)

    # Encode province
    province_encoded = encoder.transform(df_input[['province']])
    province_encoded_df = pd.DataFrame(province_encoded, columns=encoder.get_feature_names_out(['province']))

    # Combine features
    X_input = pd.concat([df_input.drop(['province', 'date', 'day_of_week'], axis=1), province_encoded_df], axis=1)

    # Ensure columns match the original training data
    X_input = X_input.reindex(columns=X_final.columns, fill_value=0)

    # Scale input data
    X_input_scaled = scaler.transform(X_input)

    # Make weather prediction
    weather_prediction = weather_model.predict(X_input_scaled)

    # Prepare input for rain model
    rain_input = pd.DataFrame({
        'max': [weather_prediction[0, 0]],
        'min': [weather_prediction[0, 1]],
        'wind': [weather_prediction[0, 2]],
        'wind_degree': [weather_prediction[0, 3]],
        'humidi': [weather_prediction[0, 4]],
        'cloud': [weather_prediction[0, 5]],
        'pressure': [weather_prediction[0, 6]]
    })

    # Scale rain input
    rain_input_scaled = scaler_rain.transform(rain_input)

    # Make rain prediction
    rain_prediction = rain_model.predict(rain_input_scaled)

    # Prepare response
    response = {
        'max_temperature': float(weather_prediction[0, 0]),
        'min_temperature': float(weather_prediction[0, 1]),
        'wind_speed': float(weather_prediction[0, 2]),
        'wind_degree': float(weather_prediction[0, 3]),
        'humidity': float(weather_prediction[0, 4]),
        'cloud': float(weather_prediction[0, 5]),
        'pressure': float(weather_prediction[0, 6]),
        'have_rain': bool(rain_prediction[0]),
    }

    # Check if it will rain and predict rain amount if necessary
    if rain_prediction[0] == 1:  # If it predicts rain
        # Scale for rain amount model
        rain_amount_input_scaled = scaler_rain_amount.transform(rain_input)
        rain_amount_prediction = rain_amount_model.predict(rain_amount_input_scaled)
        response['rain'] = float(rain_amount_prediction[0])
    else:
        response['rain'] = 0.0  # No rain expected

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5011)
