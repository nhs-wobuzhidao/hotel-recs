🏨 Hotel Recommendation System

A machine learning–based system for predicting user ratings of hotels using historical review data.

📌 Overview

This project aims to predict hotel review ratings given:

Author ID (user)
Hotel ID
Date of review

By learning patterns in user behavior and hotel characteristics, the system provides personalized rating predictions that can be used for recommendation, ranking, or user experience enhancement.

🎯 Objective

Build and evaluate models that accurately predict a user's rating for a hotel at a given point in time.

🧠 Models
🔹 Baseline Models

These serve as reference points for performance:

K-Nearest Neighbors (KNN)
Captures similarity between users or items based on past interactions.
Two-Tower Neural Network
Learns separate embeddings for users and hotels, then combines them to predict ratings.
🔹 Advanced Variants
LightGBM
Gradient boosting framework optimized for speed and performance on tabular data.
Factorization Machines (FM)
Effective for sparse data; models pairwise feature interactions (e.g., user–hotel relationships).
XGBoost
High-performance gradient boosting model with strong regularization and scalability.
📊 Features

Typical features used in the models may include:

User features (e.g., past ratings, activity level)
Hotel features (e.g., average rating, location, popularity)
Temporal features (e.g., review date, seasonality)
Interaction features (user–hotel combinations)
⚙️ Pipeline
Data Preprocessing
Handle missing values
Encode categorical variables (authorID, hotelID)
Feature engineering (time-based features, aggregates)
Model Training
Train baseline models
Train advanced models with hyperparameter tuning
Evaluation
Metrics: RMSE, MAE (or others depending on setup)
Compare baseline vs. advanced models
Prediction
Generate rating predictions for unseen user–hotel pairs
📈 Evaluation Metrics
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)

These metrics measure how close predicted ratings are to actual user ratings.

🚀 Future Improvements
Incorporate review text using NLP models
Use deep learning architectures (e.g., transformers for user behavior)
Add contextual signals (location, travel purpose)
