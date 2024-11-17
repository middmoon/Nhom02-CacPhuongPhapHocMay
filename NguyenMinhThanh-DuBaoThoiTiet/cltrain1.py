import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score
import joblib

# Load your dataset
# df should contain all necessary columns, including rain indicators and rain amount
df = pd.read_csv('your_dataset.csv')

# Define features for Model 1
X_model1 = df[['province', 'year', 'month', 'day', 'day_of_week_encoded', 'week_of_year', 'year_quarter', 'month_period']]
y_model1 = df[['max', 'min', 'wind', 'wind_degree', 'humidi', 'cloud', 'pressure']]

# OneHotEncode 'province' for Model 1
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_province_encoded = pd.DataFrame(encoder.fit_transform(X_model1[['province']]), 
                                  columns=encoder.get_feature_names_out(['province']))
X_model1_final = pd.concat([X_model1.drop('province', axis=1).reset_index(drop=True), 
                            X_province_encoded.reset_index(drop=True)], axis=1)

# Scale Model 1 features
scaler = StandardScaler()
X_model1_scaled = scaler.fit_transform(X_model1_final)

# Split data for Model 1
X_train_model1, X_test_model1, y_train_model1, y_test_model1 = train_test_split(X_model1_scaled, y_model1, test_size=0.3, random_state=42)

# Train Model 1 (Random Forest Regressor for each weather parameter)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Model 1: Multi-output Regressor
model1 = RandomForestRegressor(random_state=42)
grid_search_model1 = GridSearchCV(model1, param_grid, cv=3, n_jobs=-1)
grid_search_model1.fit(X_train_model1, y_train_model1)

# Evaluate Model 1
y_pred_model1 = grid_search_model1.predict(X_test_model1)
print("Model 1 - R^2 Score:", r2_score(y_test_model1, y_pred_model1))

# Prepare inputs for Model 2 using predictions from Model 1
X_model2 = pd.concat([X_model1, pd.DataFrame(y_pred_model1, columns=['max', 'min', 'wind', 'wind_degree', 'humidi', 'cloud', 'pressure'])], axis=1)
y_model2 = df['have_rain']  # Target for Model 2: Whether it will rain or not

# OneHotEncode and scale for Model 2
X_model2_final = pd.concat([X_model2.drop('province', axis=1).reset_index(drop=True), 
                            X_province_encoded.reset_index(drop=True)], axis=1)
X_model2_scaled = scaler.transform(X_model2_final)

# Split data for Model 2
X_train_model2, X_test_model2, y_train_model2, y_test_model2 = train_test_split(X_model2_scaled, y_model2, test_size=0.3, random_state=42)

# Model 2: Rain Classification
grid_search_model2 = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_model2.fit(X_train_model2, y_train_model2)
model2 = grid_search_model2.best_estimator_

# Evaluate Model 2
y_pred_model2 = model2.predict(X_test_model2)
print('Model 2 - Rain Classification Accuracy:', accuracy_score(y_test_model2, y_pred_model2))
print('Model 2 - F1 Score:', f1_score(y_test_model2, y_pred_model2))
print('Model 2 - Confusion Matrix:\n', confusion_matrix(y_test_model2, y_pred_model2))

# Prepare data for Model 3 (only rows where Model 2 predicted rain)
df_rain = df[df['have_rain'] == True]
X_model3 = pd.concat([X_model1, pd.DataFrame(y_pred_model1, columns=['max', 'min', 'wind', 'wind_degree', 'humidi', 'cloud', 'pressure'])], axis=1)
y_model3 = df_rain['rain_amount']  # Target for Model 3: Rain amount

# OneHotEncode and scale for Model 3
X_model3_final = pd.concat([X_model3.drop('province', axis=1).reset_index(drop=True), 
                            X_province_encoded.reset_index(drop=True)], axis=1)
X_model3_scaled = scaler.transform(X_model3_final)

# Split data for Model 3
X_train_model3, X_test_model3, y_train_model3, y_test_model3 = train_test_split(X_model3_scaled, y_model3, test_size=0.3, random_state=42)

# Model 3: Rain Amount Prediction
grid_search_model3 = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search_model3.fit(X_train_model3, y_train_model3)
model3 = grid_search_model3.best_estimator_

# Evaluate Model 3
y_pred_model3 = model3.predict(X_test_model3)
print('Model 3 - Rain Amount R^2 Score:', r2_score(y_test_model3, y_pred_model3))

# Test Prediction for all three models with a sample input
test_input = pd.DataFrame([{
    'province': 'Ho Chi Minh City', 'year': 2024, 'month': 9, 'day': 20, 
    'day_of_week_encoded': 4, 'week_of_year': 38, 'year_quarter': 3, 
    'month_period': 3
}])

# Encode province and standardize test input
test_input_encoded = pd.DataFrame(encoder.transform(test_input[['province']]), 
                                  columns=encoder.get_feature_names_out(['province']))
test_input_final = pd.concat([test_input.drop('province', axis=1).reset_index(drop=True), 
                              test_input_encoded.reset_index(drop=True)], axis=1)
test_input_scaled = scaler.transform(test_input_final)

# Model 1 Prediction: Weather Parameters
weather_params = grid_search_model1.predict(test_input_scaled)
print('Predicted Weather Parameters (Model 1):', weather_params[0])

# Model 2 Prediction: Rain Classification
rain_input = pd.concat([test_input, pd.DataFrame(weather_params, columns=['max', 'min', 'wind', 'wind_degree', 'humidi', 'cloud', 'pressure'])], axis=1)
rain_input_final = pd.concat([rain_input.drop('province', axis=1).reset_index(drop=True), test_input_encoded.reset_index(drop=True)], axis=1)
rain_input_scaled = scaler.transform(rain_input_final)
rain_prediction = model2.predict(rain_input_scaled)
print('Rain Prediction (Model 2):', 'Yes' if rain_prediction[0] else 'No')

# Model 3 Prediction: Rain Amount (if rain predicted)
if rain_prediction[0]:
    rain_amount = model3.predict(rain_input_scaled)
    print('Predicted Rain Amount (Model 3):', rain_amount[0])
else:
    print('No rain predicted, so no rain amount forecast.')
