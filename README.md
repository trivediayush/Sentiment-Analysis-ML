# Mental Health Sentiment Analysis

## Overview
This project leverages machine learning and natural language processing (NLP) techniques to classify mental health-related text into various emotional and psychological statuses. With a dataset of over 50,000 entries, this project provides insights into the potential for AI in mental health analysis.

## Features
1. Sentiment analysis of mental health.
2. Comparison of four machine learning models: 
   - Bernoulli Naive Bayes
   - Decision Tree
   - Logistic Regression
   - XGBoost
3. Visualizations:
   - Bar plot for accuracy comparison.
   - Confusion Matrix for performance analysis.
   - WordClouds for textual insights.

## Dataset Overview
- **Total Data Points**: 52,681
- **Unique Text Entries**: 51,073
- **Classes (Statuses)**: 7 (e.g., Anxiety, Depression, Normal, Bipolar, Suicidal, etc.)
- **Most Frequent Status**: Normal (16,351 instances)

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `WordCloud`
  - NLP: `nltk`, `re`, `PorterStemmer`
  - Machine Learning: `scikit-learn`, `imblearn`, `XGBoost`
- **Development Environment**: Jupyter Notebook

## Methodology
### 1. Data Preprocessing
- Tokenization and stemming using NLTK.
- Removal of special characters and stopwords.
- Balancing imbalanced data using Random Oversampling.
- Feature extraction using TF-IDF Vectorizer.

### 2. Model Training
- Classifiers used:
  - Bernoulli Naive Bayes
  - Decision Tree
  - Logistic Regression
  - XGBoost
- Performance metrics: Accuracy, Precision, Recall, and F1 Score.

### 3. Evaluation
- XGBoost emerged as the best-performing model with an accuracy of 81%.
- F1 Score for Normal status: 93
- F1 Score for Anxiety and Bipolar: 85

## Results
- **Top Performing Model**: XGBoost
- **Accuracy**: 81%
- **Insights**:
  - "Normal" state was the easiest to classify.
  - Some overlap was observed between Depression and Suicidal statuses.

### Clone the Repository
```bash
git clone https://github.com/trivediayush/Sentiment-Analysis-ML.git

