**ML Model Deployment with CI/CD Pipeline**
Project Overview

This project demonstrates the end-to-end workflow of developing, packaging, and deploying a Machine Learning model using modern DevOps practices.
The project follows the architecture:
Machine Learning Model → Flask API → Docker Container → GitHub Actions CI/CD → Google Cloud Run
In the first phase, a classification model is trained using Scikit-learn and saved for later integration with a Flask web application.

Objectives: 

Train a machine learning classification model
Evaluate model performance
Save the trained model using Joblib
Manage project code using Git and GitHub
Prepare the project for containerization and CI/CD automation

Project Structure:

ml-cloud-deployment/
│
├── train.py
├── test_model.py
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore

Dataset:
The project uses the Breast Cancer Wisconsin Dataset available in Scikit-learn.
Features:
30 numerical input features
Binary classification problem
Predicts whether a tumor is malignant or benign

Technologies Used:

Python 3.x
Scikit-learn
Pandas
NumPy
Joblib
Git & GitHub
