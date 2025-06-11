Email Spam Classification using Naive Bayes
Here's a simple example of email spam classification using Naive Bayes in Python:
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

Sample dataset
emails = [
    ("You won a prize!", 1),  # spam
    ("Meeting on Friday", 0),  # not spam
    ("Buy now and get 20% off!", 1),  # spam
    ("Project update", 0),  # not spam
    ("Make money fast!", 1),  # spam
    ("Hello, how are you?", 0),  # not spam
]

Split data into features (X) and labels (y)
X = [email[0] for email in emails]
y = [email[1] for email in emails]

Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Create a CountVectorizer object
vectorizer = CountVectorizer()

Fit the vectorizer to the training data and transform both the training and testing data
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_count, y_train)

Make predictions on the test set
y_pred = clf.predict(X_test_count)

Evaluate the model
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

Use the model to classify a new email
new_email = ["You won a prize! Claim now!"]
new_email_count = vectorizer.transform(new_email)
prediction = clf.predict(new_email_count)
print("Prediction:", prediction)
How it Works
1. The code first imports the necessary libraries, including MultinomialNB for Naive Bayes classification and CountVectorizer for converting text data into numerical features.
2. A sample dataset of labeled emails is created, where each email is marked as either spam (1) or not spam (0).
3. The data is split into training and testing sets using train_test_split.
4. A CountVectorizer object is created and fit to the training data, which converts the text data into numerical features.
5. A Multinomial Naive Bayes classifier is trained on the training data using clf.fit.
6. The model is evaluated on the test set using y_pred == y_test.
7. Finally, the model is used to classify a new email.

Advantages
- Naive Bayes is a simple and efficient algorithm for text classification tasks like email spam classification.
- It can handle high-dimensional data and is relatively fast to train.

Disadvantages
- Naive Bayes assumes independence between features, which may not always be true in practice.
- It can be sensitive to the choice of features and may not perform well with complex or noisy data.
