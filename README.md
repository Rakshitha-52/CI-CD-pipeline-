ML Model Deployment with CI/CD Pipeline
Project Overview
This project demonstrates the end-to-end workflow of developing, packaging, and deploying a Machine Learning model using modern DevOps practices.
The project follows the architecture:
Machine Learning Model → Flask API → Docker Container → GitHub Actions CI/CD → Google Cloud Run
In the first phase, a classification model is trained using Scikit-learn and saved for later integration with a Flask web application.

Objectives
Train a machine learning classification model
Evaluate model performance
Save the trained model using Joblib
Manage project code using Git and GitHub
Prepare the project for containerization and CI/CD automation
