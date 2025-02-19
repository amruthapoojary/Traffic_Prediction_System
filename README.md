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
git clone https://github.com/amruthapoojary/Traffic_Prediction_System
cd Traffic_Prediction_System
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
