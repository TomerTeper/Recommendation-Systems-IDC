# Recommendation-Systems-IDC

## Overview

Recommendation systems are algorithms that help users discover relevant items based on their preferences and behaviors. These systems are widely used in domains like e-commerce, streaming platforms, and online advertising. They can be categorized into Collaborative Filtering (CF), Content-Based Filtering, Hybrid Methods, and Deep Learning-based approaches.

This document consolidates key concepts from multiple assignments and the final project in the Recommendation Systems course, covering evaluation metrics, matrix factorization, and model-based collaborative filtering.

## Table of Contents

1. **Homework 1 - Evaluation Metrics**
2. **Homework 2 - Matrix Factorization**
3. **Final Project - MovieLens 100K Recommendation System**

---

## **1. Homework 1 - Evaluation Metrics**

### Overview

Focuses on evaluating recommender systems using the MovieLens 100K dataset and various metrics.

### Key Topics Covered

- RMSE, MAE, Precision@K, Recall@K, NDCG
- Coverage and diversity analysis
- Comparison of evaluation metrics and their biases

### Implementation & Insights

- Explored rating distributions and user behaviors
- Computed multiple evaluation metrics
- Discussed trade-offs between error-based and ranking-based evaluation methods

---

## **2. Homework 2 - Matrix Factorization**

### Overview

Explores matrix factorization techniques for recommendation models, focusing on implementation, optimization, and evaluation.

### Key Topics Covered

- Batch Gradient Descent (GD) & Alternating Least Squares (ALS)
- Hyperparameter tuning and model comparison
- Overfitting, underfitting, and item similarity analysis

### Implementation & Insights

- Implemented MF models (GD, ALS) using NumPy & Pandas
- Tuned hyperparameters to optimize performance
- Compared model accuracy, training time, and generalization

### Future Enhancements

- Hybrid models combining CF and content-based filtering
- Neural Collaborative Filtering (NCF)
- Cold-start solutions using metadata

---

## **3. Final Project - MovieLens 100K Recommendation System**

### Overview

Developed a recommendation system using collaborative filtering, matrix factorization, and explainability techniques.

### Key Topics Covered

- Exploratory Data Analysis (EDA)
- User-based & item-based CF
- Deep Learning-based MF
- Model-based CF (XGBoost, LightGBM)
- Explainable AI (SHAP) & W-RMSE evaluation

### Insights & Results

- CF models performed well but faced sparsity issues
- Deep learning MF improved accuracy but was computationally intensive
- Model-based CF outperformed traditional CF
- SHAP analysis provided feature importance insights

### Future Work

- Enhancing deep learning models with attention mechanisms
- Expanding to larger datasets (MovieLens 1M/10M)
- Improving explainability using LIME & SHAP

### Python Libraries Used

- **Pandas, NumPy**: Data manipulation
- **Matplotlib, Seaborn**: Visualization
- **Scikit-learn**: ML models
- **Surprise**: CF models
- **XGBoost, LightGBM**: Model-based CF
- **TensorFlow, PyTorch**: Deep learning
- **SHAP**: Explainability

