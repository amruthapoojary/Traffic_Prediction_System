# Traffic_Prediction_System

## Description
This is a web-based application that predicts traffic situations based on vehicle counts (car, bike, bus, truck), the time of day, and other features. The application uses a Random Forest Classifier machine learning model trained on a traffic dataset. The result of the prediction is displayed along with a bar graph of vehicle counts.

## Features
- Predicts traffic situations (e.g., Low, Medium, High) based on input features.
- Visualizes the vehicle count breakdown for different types of vehicles (Car, Bike, Bus, Truck).
- Interactive web interface built with Flask.

## Technologies Used
- **Backend**: Flask (Python)
- **Machine Learning**: Random Forest Classifier (Scikit-learn)
- **Data Visualization**: Matplotlib
- **Frontend**: HTML, CSS

## Setup and Installation

### 1. Clone the Repository
Clone this project to your local machine:

git clone https://github.com/your-username/traffic-prediction.git
text

### 2. Install Dependencies

Navigate to the project directory and create a virtual environment:

cd traffic-prediction
python -m venv venv
text

Activate the virtual environment:

**Windows:**

venv\Scripts\activate
text

**Linux/MacOS:**

source venv/bin/activate
text

Install the required dependencies from the `requirements.txt` file:

pip install -r requirements.txt
text

### 3. Run the Application

After installing the dependencies, run the Flask application:

python app.py
text

### 4. Access the Application

Open your browser and go to `http://127.0.0.1:5000/` to interact with the application.

## Screenshots

**Home Page (Form):**
  *(Image of the Home Page Form)*

**Prediction Result:**
  *(Image of the Prediction Result)*

## License

This project is open-source and available under the MIT License.
# 🚦 Traffic Prediction Web Application

## Overview
This is a *real-time traffic prediction system* designed to forecast traffic situations based on various factors like *vehicle counts, time of day, and day of the week*. The application uses a *Random Forest Classifier* machine learning model to predict the traffic condition (e.g., Low, Medium, High) and visualize the breakdown of different types of vehicles.

## Technologies Used
- **Frontend:** HTML, CSS
- **Backend:** Flask (Python)
- **Machine Learning:** Random Forest Classifier (Scikit-learn)
- **Data Visualization:** Matplotlib
- **Database:** Not used in this version

## Installation & Setup

### 1️⃣ Clone the Repository
Clone the project repository to your local machine:
```bash
git clone https://github.com/your-username/traffic-prediction.git
cd traffic-prediction
```
## Installation & Setup

### 2️⃣ Set Up Virtual Environment
Create a virtual environment to manage dependencies:
```bash
python -m venv venv
```
 Activate the virtual environment:

Windows:
```bash
venv\Scripts\activate
```

Linux/MacOS:
```bash
source venv/bin/activate
```
### 3️⃣ Install Dependencies
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 4️⃣ Start the Flask Backend Server

Run the following command to start the backend server:
```bash
python app.py
```
 This will run the Flask application on 'http://localhost:5000'.

 Access the Application
Open your browser and navigate to 'http://127.0.0.1:5000/' to interact with the application.

## Features
✔️ **Real-Time Traffic Prediction** - Predicts traffic conditions based on user inputs.  
✔️ **Vehicle Count Breakdown** - Visualizes the vehicle counts (Cars, Bikes, Buses, Trucks).  
✔️ **Interactive User Interface** - Users can input data like time, vehicle count, and day of the week to get predictions.  
✔️ **Error Handling** - Displays appropriate error messages if any issue occurs during form submission or data processing.

## Screenshots
### Homepage View (Form)
![App Screenshot](An1.png)

![App Screenshot](An2.png)

### Prediction Result
![App Screenshot](An3.png)

## License
This project is licensed under the MIT License.
