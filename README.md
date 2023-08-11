# NLP Project: Sentiment Analysis of Amazon Alexa Reviews

### Overview:
Our NLP project focuses on the sentiment analysis of customer reviews about Amazon Alexa. The goal is to build an artificial intelligence (AI) model to predict sentiment levels from customer reviews of Amazon Echo products. By automating the sentiment analysis process, we want to provide valuable information on customer satisfaction.

### Target:

- Build and train an AI model to analyze the emotions of Amazon Alexa reviews.
- Categorize reviews into positive, negative, or neutral emotional categories.
- Provide an automated tool to predict customer satisfaction with products.
- Facilitate the efficient analysis of a large number of customer reviews.

### Tools and Technology:

- Google Colab notebooks for development and testing.
- Python programming language.
- Scikit-learn library to implement NLP algorithms and build emotional analysis models.
- Matplotlib and Seaborn libraries for data visualization.

### Data sources:
The data for this project was collected from Kaggle, including real customer [reviews of Amazon Alexa Echo products](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews). This dataset provides a range of different emotions expressed by customers.

### Expected results:
I aim to build an accurate and reliable AI model to analyze emotions in Amazon Alexa reviews. This model automatically categorizes reviews into positive, negative, or neutral categories, helping to provide helpful information for decisions within the company. And this project will bring positive contributions and help improve the efficiency of customer reviews.

## Key Findings and Results:

### Model: Naive Bayes Classifier

```python
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# ... (Prediction, Confusion Matrix, Classification Report)
```
- Achieved an accuracy of approximately 93%.
- Notable precision and recall for positive sentiment (class 1), with a minor trade-off in recall for negative sentiment (class 0).
### Model: Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# ... (Prediction, Confusion Matrix, Classification Report)
```
- Attained an accuracy of approximately 93%.
- Demonstrated proficient recall and precision for positive sentiment (class 1), while experiencing a shortfall in recall for negative sentiment (class 0).

### Model: Gradient Boosting Classifier
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# ... (Prediction, Confusion Matrix, Classification Report)
```
- Achieved an accuracy of around 91%.
- Notable precision and recall for the positive sentiment (class 1), yet faced challenges in recall for the negative sentiment (class 0).

## Conclusions
The developed models exhibit promising results in predicting customer sentiment within Amazon Alexa reviews. While each model demonstrates commendable accuracy, their individual strengths and weaknesses are evident in terms of identifying negative sentiment. Further optimization and model tuning can potentially enhance the overall performance and equilibrium between the two sentiment categories.

This project significantly contributes to the realm of customer satisfaction analysis, showcasing the potential of NLP and AI to automate and streamline the evaluation of sentiments expressed by customers. Through this endeavour, we aim to empower decision-making processes within the company and ultimately enhance customer experience.
