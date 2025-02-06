from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os


app = Flask(__name__)


# Load the dataset
dataset = pd.read_csv('Traffic.csv')

# Data Preprocessing
def preprocess_data(df):
    df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour
    df['Minute'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.minute
    df['DayOfWeek'] = df['Day of the week'].astype('category').cat.codes
    df['TimePeriod'] = df['Hour'].apply(lambda hour: 'Morning' if 5 <= hour < 12 else
                                                    'Afternoon' if 12 <= hour < 17 else
                                                    'Evening' if 17 <= hour < 21 else 'Night')
    df['TimePeriod'] = df['TimePeriod'].astype('category').cat.codes
    return df

dataset = preprocess_data(dataset)

# Splitting Features and Target
features = ['Hour', 'Minute', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'DayOfWeek', 'Date', 'TimePeriod']
target = 'Traffic Situation'

X = dataset[features]
y = dataset[target].astype('category').cat.codes

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    try:
        input_data = np.array([[
            int(form_data['hour']),
            int(form_data['minute']),
            int(form_data['car_count']),
            int(form_data['bike_count']),
            int(form_data['bus_count']),
            int(form_data['truck_count']),
            int(form_data['day_of_week']),
            int(form_data['day_of_month']),
            int(form_data['time_period'])
        ]])

        # Predict using the model
        prediction = model.predict(input_data)[0]
        traffic_situation = dataset['Traffic Situation'].astype('category').cat.categories[prediction]

        # Generate a simple graph
        fig, ax = plt.subplots()
        ax.bar(['Car', 'Bike', 'Bus', 'Truck'], input_data[0][2:6], color=['blue', 'green', 'orange', 'red'])
        ax.set_title('Vehicle Count Breakdown')
        plt.savefig('static/prediction_graph.png')
        plt.close()

        return render_template('result.html', prediction=traffic_situation)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)