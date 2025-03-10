
# FAKE NEWS DETECTION


## ABSTRACT

The spread of fake news is a major concern in today's digital world. This project aims to build a machine learning model to classify news articles as fake or real using Natural Language Processing (NLP) techniques. The approach involves text preprocessing, feature extraction using TF-IDF, and training a Naïve Bayes classifier to make predictions. The model's performance is evaluated using accuracy, confusion matrix, and ROC curve.



## OVERVIEW

 This project processes and classifies news articles based on their authenticity. The steps include:

-Data Cleaning & Preprocessing – Removing noise, stopwords, and formatting text.
-Feature Extraction – Applying TF-IDF vectorization with bigrams to convert text into numerical form.
-Model Training – Using Multinomial Naïve Bayes for text classification.
-Evaluation – Analyzing classification metrics, feature importance, and visualizations.
=Prediction – Implementing a function to classify new text as fake or real.


## Project Goals

-Develop a machine learning model to detect fake news.
-Improve classification accuracy using TF-IDF and Naïve Bayes.
-Analyze the most important features that indicate fake news.
-Provide a simple function for real-time news classification.


## DATA SET
The dataset used in this project consists of two CSV files:
-Fake.csv – Contains text from fake news articles labeled as 1.
-True.csv – Contains text from real news articles labeled as 0.
The dataset is shuffled and split into training (80%) and testing (20%) sets. The text data is preprocessed to remove unnecessary symbols, links, and punctuation.

## Data Source        
[Fake News Dataset](https://www.kaggle.com/datasets/jainpooja/fake-news-detection)


## Algorithm

-Data Preprocessing – Cleaning text by removing punctuation, stopwords, and special characters.
-Feature Extraction – Converting text into numerical representation using TF-IDF vectorization.
-Model Selection – Training a Multinomial Naïve Bayes classifier, a probabilistic model well-suited for text classification.
-Evaluation Metrics:
    -Accuracy – Measures overall model performance.
    -Confusion Matrix – Identifies classification errors.
    -ROC Curve & AUC Score – Evaluates the model’s discrimination ability.

## Conclusion

  The implemented Naïve Bayes classifier with TF-IDF achieves promising results in detecting fake news. The model effectively identifies key indicators of fake news and provides high accuracy with a simple yet efficient approach. Future improvements could involve deep learning models such as LSTMs or transformers to enhance accuracy and handle complex linguistic patterns.



