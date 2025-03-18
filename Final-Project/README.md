# **Recommendation System Final Project - MovieLens 100K**

## Overview

This project focuses on building a recommendation system using the MovieLens 100K dataset. The dataset contains user-movie interactions and ratings, allowing for the development of various collaborative filtering (CF) and machine learning-based recommendation models. This project emphasizes Exploratory Data Analysis (EDA), Collaborative Filtering (CF) approaches, Matrix Factorization (MF) using Deep Learning, Model-based Collaborative Filtering (MCF), Explainable AI (XAI), and Weighted RMSE (W-RMSE) as a primary evaluation metric.

## Table of Contents

1. **Dataset Overview**
2. **Exploratory Data Analysis (EDA)**
3. **Collaborative Filtering (CF) Approaches**
4. **Matrix Factorization (MF) using Deep Learning**
5. **Model-based Collaborative Filtering (MCF)**
6. **Explainable AI (XAI) in Recommendation Systems**
7. **Evaluation Metrics - W-RMSE**
8. **Results & Insights**
9. **Future Work**
10. **Python Libraries Used**

## 1. Dataset Overview

The dataset contains three main files:

- **Train Dataset**: Contains user ratings for movies.
  - `userId`: Unique identifier for users.
  - `movieId`: Unique identifier for movies.
  - `rating`: Ratings given by users to movies.
- **Movies Dataset**: Contains movie details.
  - `movieId`: Unique identifier for movies.
  - `title`: Movie title.
  - `genres`: Movie genres.
- **Tags Dataset**: Contains additional contextual information about movies.
  - `userId`: Unique identifier for users who assigned tags.
  - `movieId`: Unique identifier for movies associated with the tag.
  - `tag`: User-defined tags describing movie characteristics (e.g., "music", "bullying", "weird").

### Insights:

- The dataset has over 100,000 ratings.
- The average rating is approximately 3.61 with a standard deviation of 1.02.
- Ratings range from 0.5 to 5.0.
- User-defined tags provide additional contextual information that can be leveraged for content-based recommendations.

## 2. Exploratory Data Analysis (EDA)

- Analyzing rating distributions, user activity, and movie popularity.
- Identifying biases in rating distributions (e.g., popular movies receiving higher ratings).
- Visualizing relationships between user engagement and rating patterns.

## 3. Collaborative Filtering (CF) Approaches

- **User-Based CF**: Finding similar users and recommending items based on their preferences.
- **Item-Based CF**: Finding similar items and recommending them based on user interactions.
- **Similarity Metrics Used**: Pearson correlation, Cosine similarity.

## 4. Matrix Factorization (MF) using Deep Learning

- Implementing Singular Value Decomposition (SVD), Alternating Least Squares (ALS), and Non-Negative Matrix Factorization (NMF).
- Deep Learning-based matrix factorization models using Neural Networks.
- Optimizing embeddings for users and items to improve prediction accuracy.

## 5. Model-based Collaborative Filtering (MCF)

- Using advanced machine learning models like XGBoost, LightGBM, and deep learning for user-movie interactions.
- Feature engineering techniques to enhance model performance.
- Hyperparameter tuning and cross-validation strategies.

## 6. Explainable AI (XAI) in Recommendation Systems

- Utilizing **SHAP (SHapley Additive exPlanations)** to interpret model predictions.
- Analyzing SHAP values to identify key features that contribute to rating predictions.
- Enhancing model transparency to improve user trust and debugging.

## 7. Evaluation Metrics - W-RMSE

The primary evaluation metric used in this project is **Weighted RMSE (W-RMSE)**, which penalizes incorrect predictions more significantly for higher ratings.

- RMSE formula: \(RMSE = \sqrt{\frac{\sum (y_{true} - y_{pred})^2}{n}}\)
- Weighted RMSE adjusts this by incorporating rating frequency-based weights.

## 8. Results & Insights

- **Collaborative Filtering** models provided reasonable recommendations but suffered from sparsity issues.
- **Matrix Factorization with Deep Learning** improved accuracy but required significant computational power.
- **MCF (Model-based CF)** outperformed traditional CF approaches due to feature-rich representations.
- **XAI using SHAP** provided valuable insights into feature importance in recommendation model decisions.
- **W-RMSE** demonstrated that considering rating importance improves evaluation robustness.

## 9. Future Work

- Enhancing deep learning models with attention mechanisms.
- Integrating content-based filtering for hybrid recommendations.
- Expanding the approach to larger datasets like MovieLens 1M or 10M.
- Further improvements in explainability using LIME and SHAP.

## 10. Python Libraries Used

The following Python libraries were used in this project:

- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical operations and array computations.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning models, preprocessing, and evaluation metrics.
- **Surprise**: Implementation of collaborative filtering and matrix factorization models.
- **XGBoost & LightGBM**: Gradient boosting models for model-based collaborative filtering.
- **TensorFlow & PyTorch**: Deep learning frameworks for building neural network-based models.
- **SHAP**: Explainability framework for interpreting machine learning models.
