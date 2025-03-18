# Recommender Systems - Homework 2

## Matrix Factorization

## Overview

In this assignment, we focus on implementing, optimizing, and evaluating recommender system models based on matrix factorization. We will compare different algorithms, fine-tune hyperparameters, and analyze performance using various metrics. Finally, we will assess the qualitative effectiveness of the models through item similarity analysis.

## Key Topics Covered

- Matrix Factorization using Batch Gradient Descent (GD) and Alternating Least Squares (ALS)
- Model training and evaluation
- Hyperparameter tuning using grid search
- Comparison of different recommender models (Popularity, Bias, ALS, GD)
- Overfitting and underfitting analysis
- Item similarity and qualitative assessment of recommendations

## Python and Libraries Used

This assignment requires the use of Python and several key libraries:

- **Python version**: 3.x
- **Libraries**:
  - `numpy` - for numerical computations
  - `pandas` - for data manipulation and analysis
  - `matplotlib` & `seaborn` - for visualization
  - `sklearn.metrics` - for model evaluation

## Implementation Tasks

### 1. Data Preparation

- Load the dataset and preprocess the rating matrix.
- Handle missing ratings appropriately.
- Consider user and item metadata but focus primarily on rating data.

### 2. Matrix Factorization Models

#### Implement the following MF algorithms using NumPy and Pandas:

- **Batch Gradient Descent (GD) Model**
- **Alternating Least Squares (ALS) Model**

### 3. Model Training and Evaluation

- Train all models (Popularity, Bias, GD, ALS) using default hyperparameters.
- Compare models using various evaluation metrics.

### 4. Hyperparameter Tuning

- Perform grid search to optimize hyperparameters for ALS and GD.
- Discuss best configurations and insights into overfitting and underfitting.

### 5. Model Comparison and Insights

- Analyze performance in terms of accuracy, training time, and generalization.
- Discuss scenarios where different models are more suitable.

### 6. Model Enhancements (Bonus)

- Suggest possible improvements such as incorporating user/item features, advanced factorization techniques, and deep learning approaches.

### 7. Item Similarity and Explainability

- Implement an item similarity method using NumPy and Pandas.
- Compare recommendations from different models for selected movies.

## Observations and Insights

- **Popularity Model**: Fast but lacks personalization.
- **Bias Model**: Incorporates user and item biases, improving accuracy slightly.
- **ALS and GD Models**: Provide the best accuracy, but GD has higher training time.
- **Hyperparameter Impact**:
  - More latent factors improve accuracy but may lead to overfitting.
  - Regularization prevents overfitting but excessive values can cause underfitting.
  - Learning rate in GD must be carefully tuned for convergence.

## Future Enhancements

- **Hybrid models** combining collaborative filtering with content-based approaches.
- **Deep learning methods** such as Neural Collaborative Filtering (NCF).
- **Cold-start solutions** using metadata-based recommendations.
- **Explainable AI techniques** to justify recommendations.
