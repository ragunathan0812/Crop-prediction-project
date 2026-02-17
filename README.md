🌾 AI-Based Crop Recommendation System

A Machine Learning–powered application that recommends the most suitable crop based on soil nutrients and climatic conditions.

Built using Python, Scikit-learn, and Streamlit.

📌 Problem Statement

Choosing the right crop based on soil and climate conditions is crucial for maximizing agricultural productivity. This project uses machine learning to analyze soil nutrients and environmental factors to recommend the optimal crop.

📊 Dataset Overview

The dataset contains:

Nitrogen (N)

Phosphorus (P)

Potassium (K)

Temperature (°C)

Humidity (%)

pH

Rainfall (mm)

Target: Crop Label

Total Samples: 2200

Multiple crop classes

Balanced dataset

🔎 Exploratory Data Analysis (EDA)

The notebook includes comprehensive analysis:

1️⃣ Nutrient Analysis by Crop

📌 Insight: Different crops require significantly different nitrogen levels.

<img width="1389" height="590" alt="Average Nitrogen content for each crop" src="https://github.com/user-attachments/assets/64499545-48d8-4be2-9933-b54a6090a218" />


2️⃣ Rainfall Distribution Across Crops

📌 Insight: Rainfall is one of the strongest differentiators among crops.

<img width="1389" height="590" alt="Rainfall Distribution across Crops" src="https://github.com/user-attachments/assets/4d5249ab-a572-4e33-81f4-12ab0fe70746" />


3️⃣ Feature Distribution

📌 Insight: Features show varied distributions, some skewed.

<img width="1490" height="985" alt="Feature Distribution Analysis" src="https://github.com/user-attachments/assets/906ccd08-0aee-4577-8be4-a095f20eb848" />


4️⃣ Correlation Heatmap

📌 Insight:

Low multicollinearity

Features are largely independent

<img width="892" height="789" alt="Feature Correlation Matrix" src="https://github.com/user-attachments/assets/1f5916cd-4d10-4f7f-b036-11763b5f7d23" />


5️⃣ Temperature vs Humidity Analysis

📌 Insight:
Environmental interaction plays a key role in classification.

<img width="1190" height="790" alt="Temperature vs Humidity by Crop Type" src="https://github.com/user-attachments/assets/85a8322b-7e37-4ec8-80bf-c884a956726c" />


🤖 Machine Learning Models Evaluated

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine

🏆 Final Model: Random Forest Classifier
📈 Performance:

Accuracy: ~98–99%

Cross-validation applied

Stable across folds

🔍 Feature Importance:

Most influential features:

Rainfall

Humidity

Potassium (K)

<img width="794" height="659" alt="Feature Importance Distribution" src="https://github.com/user-attachments/assets/c4f4e94a-e61e-4ae1-a797-d565f73b0a4a" />


🌐 Web Application (Streamlit)

The system includes an interactive web app where users can:

Input soil nutrient values

Provide climate parameters

Get crop recommendation

View confidence score

See top 3 predictions

View feature importance graph

🛠 Tech Stack

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Streamlit

Joblib

📁 Project Structure
Crop-prediction-project/
│── app.py
│── notebook.ipynb
│── dataset.csv
│── models/
│── images/
│── README.md
│── requirements.txt
│── .gitignore

🚀 How to Run
git clone https://github.com/ragunathan0812/Crop-prediction-project.git
cd Crop-prediction-project

conda create -n ds python=3.10
conda activate ds
pip install -r requirements.txt

streamlit run app.py

📌 Future Improvements

Fertilizer recommendation

Crop yield prediction

Real-time weather API integration

Cloud deployment

👨‍💻 Author

Ragunathan
Machine Learning Enthusiast


