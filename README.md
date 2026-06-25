# 🚦 Road Accident Severity Prediction System

## 📌 Project Overview

The Road Accident Severity Prediction System is a Machine Learning-based web application developed using Python and Streamlit. The system predicts the severity of road accidents based on various factors such as driver age, speed limit, weather conditions, road type, traffic conditions, and alcohol involvement.

The project compares multiple machine learning algorithms and selects the best-performing model for accident severity prediction.

---

##  Objectives

* Predict accident severity levels (Low, Medium, High).
* Analyze accident-related factors.
* Compare machine learning algorithms.
* Visualize model performance using graphs and confusion matrices.
* Provide an interactive Streamlit web application for users.

---

##  Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

---

## Project Structure

```text
Road_Accident_Project/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── dataset/
│   └── karnataka_accident_dataset_4000.xlsx
│
├── models/
│   ├── trained_model.pkl
│   └── label_encoders.pkl
│
├── graphs/
│   ├── model_comparison.png
│   ├── metrics_comparison.png
│   └── confusion_matrix.png
│
└── README.md
```

---

## Machine Learning Algorithms Used

1. Decision Tree Classifier
2. Logistic Regression
3. K-Nearest Neighbors (KNN)

The models are trained and evaluated using:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Model Evaluation

The project compares the performance of all models and automatically selects the best-performing model based on accuracy.

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

##  How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/road-accident-severity-prediction.git
```

### Step 2: Navigate to Project Directory

```bash
cd road-accident-severity-prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python train_model.py
```

### Step 5: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

---

##  Application Features

### Prediction Module

Users can provide:

* Driver Age
* Speed Limit
* Weather Conditions
* Road Type
* Traffic Density
* Alcohol Involvement

The application predicts the accident severity level.

### Analytics Dashboard

Displays:

* Model Accuracy Comparison
* Precision, Recall, and F1 Score Comparison
* Confusion Matrix
* Best Performing Model

---

## Future Enhancements

* Integration with real-time accident datasets.
* Deployment on Streamlit Cloud.
* Deep Learning-based prediction models.
* Interactive data visualizations.
* Real-time accident risk monitoring.

---

##  Author

**Jeevitha**

Machine Learning  Enthusiast

---

## License

This project is developed for educational and academic purposes.
