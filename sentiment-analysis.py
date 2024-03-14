import pandas as pd
import numpy as np
import math
import string
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

import warnings
warnings.filterwarnings("ignore")
# use WIDER CANVAS:
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# import the data
products = pd.read_csv('./data/amazon_baby.gz')
print('There are',len(products),'reviews of baby products')

# replace punctuation symbols by blanks in the text
def remove_punctuation(text):
    if not isinstance(text, str):
        # If not, return an empty string or some other placeholder
        return ""
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)

review_no_punctuation = products['review'].apply(remove_punctuation)

# create the feature matrix
stop_words = ['you', 'he', 'she', 'they', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'is', 'was', 'were', 'am', 'are']

products['review'].fillna('', inplace=True)

cv = CountVectorizer(stop_words=stop_words, min_df=2, max_df=0.6)

word_count_vector = cv.fit_transform(products['review'])
products['word_count_vec'] = [sparse.csr_matrix(word_count_vector[i]) for i in range(word_count_vector.shape[0])]

# print(products['word_count_vec'][0])

# find the size of the feature vectors
vector_size = len(cv.vocabulary_)
print(f"Size of the feature vectors: {vector_size}")

# ignore all neutral sentiment reviews, where rating is equal to 3
products = products[products['rating'] != 3]

# create a sentiment column and consider reviews with a rating of 4 or higher as +1 and a rating of 2 or lower as -1
products['sentiment'] = products['rating'].apply(lambda x: 1 if x >= 4 else (-1 if x <= 2 else 0))

# split the data into training and test sets
train_data = products.sample(frac=.8, random_state=0)
test_data = products.drop(train_data.index)
print(f'We will use N={len(train_data)} training samples')
print(f'and {len(test_data)} testing samples')
print(f'Total samples: {len(products)}')

print("")
print("Training the model...")
# create a logistic regression model with scikit-learn, L2 regularization and C=100 penalty
X_train = vstack(train_data['word_count_vec'].values)

y_train = train_data['sentiment'].values

sentiment_model = LogisticRegression(C=100, penalty='l2', solver='lbfgs', max_iter=1000)
sentiment_model.fit(X_train, y_train)

print("Model trained.")
# extract the weights of the words and store them in a dictionary that maps feature names to coefficients
word_coef = {}
feature_names = cv.get_feature_names_out()
coefficients = sentiment_model.coef_[0]
word_coef = dict(zip(feature_names, coefficients))

# make predictions on sample test data
sample_test_data = test_data.iloc[90:98]
scores = []

X_sample_test = vstack(sample_test_data['word_count_vec'].values)

scores = sentiment_model.decision_function(X_sample_test)

for i, score in enumerate(scores):
    print(f"\nReview {i+1}:")
    print("Rating:", sample_test_data.iloc[i]['rating'])
    print("Review:\n", sample_test_data.iloc[i]['review'])
    print("Score (z):", score)

    true_sentiment = '+1' if sample_test_data.iloc[i]['rating'] >= 4 else '-1'
    predicted_sentiment = '+1' if score > 0 else '-1'
    print("True Sentiment:", true_sentiment)
    print("Predicted Sentiment:", predicted_sentiment)

compatible_scores_sample_test = sum((score > 0 and sample_test_data.iloc[i]['rating'] >= 4) or
                        (score <= 0 and sample_test_data.iloc[i]['rating'] <= 2)
                        for i, score in enumerate(scores))

print(f"\nNumber of scores compatible with the true sentiment on the sample test data: {compatible_scores_sample_test}")
print("")

# compute the class predictions (y hat)
y_hat_sample_test = []

for score in scores:
    if score > 0:
        y_hat_sample_test.append(+1)
    else:
        y_hat_sample_test.append(-1)

print("Class predictions (y_hat):")
print(y_hat_sample_test)

# print the sklearn class predictions to check
print("Class predictions according to sklearn:")
print(sentiment_model.predict(vstack(sample_test_data['word_count_vec'].values)))

print("")

# compute the probability that a sentiment is positive for each sample
probabilities = 1 / (1 + np.exp(-scores))
print("Probabilities of positive sentiment on the sample test data:")
print(probabilities)

# identify the review classified as positive with the lowest probability
# and check if its prediction was correct
min_positive_prob = min(prob for prob in probabilities if prob > 0.5)
lowest_prob_index = np.argmin([prob if prob > 0.5 else 1 for prob in probabilities])
print(f"Review classified as positive with the lowest probability in sample test data ({min_positive_prob}):")
print(sample_test_data.iloc[lowest_prob_index]['review'])
print("Rating:", sample_test_data.iloc[lowest_prob_index]['rating'])

print("")

# compute the sklearn sample probabilities
probabilities_sklearn = sentiment_model.predict_proba(X_sample_test)

positive_probabilities_sklearn = probabilities_sklearn[:, 1]

print("Probabilities from sklearn's predict_proba:")
print(positive_probabilities_sklearn)

print("Manual probabilities for comparison:")
print(probabilities)

are_matching = np.allclose(positive_probabilities_sklearn, probabilities, atol=1e-8)
print(f"Do the probability predictions match? {are_matching}")

print("")

# examine the whole test dataset
X_test = vstack(test_data['word_count_vec'].values)
y_test = test_data['sentiment'].values

# compute the accuracy of the classifier
def get_classification_accuracy(model, data, true_labels):
    """
    Compute the accuracy of the model on the given data and labels.
    """
    predictions = model.predict(data)
    correct_predictions = sum(predictions == true_labels)
    accuracy = correct_predictions / float(len(true_labels))
    
    return accuracy

training_accuracy = get_classification_accuracy(sentiment_model, X_train, y_train)
test_accuracy = get_classification_accuracy(sentiment_model, X_test, y_test)

print(f"Training Accuracy: {round(training_accuracy, 4)}")
print(f"Test Accuracy: {round(test_accuracy, 4)}")

print("")

# find 10 most positive and 10 most negative weights in the learned model
feature_names = cv.get_feature_names_out()
coefficients = sentiment_model.coef_[0]

features_coefficients = list(zip(feature_names, coefficients))

most_positive_features = sorted(features_coefficients, key=lambda x: x[1], reverse=True)[:10]
most_negative_features = sorted(features_coefficients, key=lambda x: x[1])[:10]

print("10 Most Positive Weights:")
for feature, coef in most_positive_features:
    print(f"{feature}: {coef}")

print("\n10 Most Negative Weights:")
for feature, coef in most_negative_features:
    print(f"{feature}: {coef}")

print("")


# learn another classifier with fewer words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves','wonderfully','lifesaver',
      'well', 'broke', 'less', 'waste', 'disappointed', 'unusable',
      'work', 'money', 'return']

# create a new CountVectorizer with only the significant words
small_vectorizer = CountVectorizer(vocabulary=significant_words)

subset_word_count_vec = small_vectorizer.fit_transform(products['review'].fillna(''))

products['word_count_subset_vec'] = list(subset_word_count_vec)

# print(products.head())

# add the new column to the training and testing dataframe
train_data['word_count_subset_vec'] = products['word_count_subset_vec']
test_data['word_count_subset_vec'] = products['word_count_subset_vec']

print("Training the simple model...")

# train a logistic regression model on the subset of words
X_train_subset = vstack(train_data['word_count_subset_vec'].values)
y_train = train_data['sentiment'].values

simple_model = LogisticRegression(C=100, penalty='l2', solver='lbfgs', max_iter=1000)
simple_model.fit(X_train_subset, y_train)

print("Simple model trained.")

print("")

# extract the weights of the words and store them in a dictionary that maps feature names to coefficients
feature_names_small_vec = small_vectorizer.get_feature_names_out()

coefficients_small_vec = simple_model.coef_[0]

intercept_small_vec = simple_model.intercept_[0]

word_coef = dict(zip(feature_names_small_vec, coefficients_small_vec))
word_coef['intercept'] = intercept_small_vec

for word, coef in sorted(word_coef.items(), key=lambda item: item[1], reverse=True):
    print(f"{word}: {coef}")

print("")

# find the number of positive coefficients
num_positive_coefficients = sum(coef > 0 for coef in coefficients_small_vec)

# determine whether the positive words in the simple model are also positive in the sentiment model
positive_significant_words = [word for word, coef in word_coef.items() if coef > 0 and word != 'intercept']

sentiment_feature_names = cv.get_feature_names_out()
sentiment_coefficients = dict(zip(sentiment_feature_names, sentiment_model.coef_[0]))

all_positive_in_sentiment_model = all(sentiment_coefficients[word] > 0 for word in positive_significant_words)

print("Are all positive significant words from simple_model also positive in sentiment_model?", all_positive_in_sentiment_model)

print("")
# compute the classification accuracies for both models on the train data
training_accuracy_sentiment_model = get_classification_accuracy(sentiment_model, X_train, y_train)

training_accuracy_simple_model = get_classification_accuracy(simple_model, X_train_subset, y_train)

print(f"Training Accuracy of sentiment_model: {training_accuracy_sentiment_model}")
print(f"Training Accuracy of simple_model: {training_accuracy_simple_model}")

higher_accuracy_model = "sentiment_model" if training_accuracy_sentiment_model > training_accuracy_simple_model else "simple_model"
print(f"Model with higher accuracy on the training set: {higher_accuracy_model}")

print("")
# compute the classification accuracies for both models on the test data
X_test_simple = vstack(test_data['word_count_subset_vec'].values)

test_accuracy_sentiment_model = get_classification_accuracy(sentiment_model, X_test, y_test)

test_accuracy_simple_model = get_classification_accuracy(simple_model, X_test_simple, y_test)

print(f"Test Accuracy of sentiment_model: {test_accuracy_sentiment_model}")
print(f"Test Accuracy of simple_model: {test_accuracy_simple_model}")

higher_accuracy_model_test = "sentiment_model" if test_accuracy_sentiment_model > test_accuracy_simple_model else "simple_model"
print(f"Model with higher accuracy on the testing set: {higher_accuracy_model_test}")

print("")
# function to return a majority classifier for the column label of frame data
def compute_majority_classifier(data, label):
    majority_label = data[label].mode()[0]
    
    def majority_classifier(_):
        return majority_label
    
    return majority_classifier

majority_classifier_train = compute_majority_classifier(train_data, 'sentiment')

majority_class = majority_classifier_train(None)
print(f"The majority classifier always predicts: {majority_class}")

# compute the accuracy of the majority classifier on the test data
majority_label = train_data['sentiment'].mode()[0]

majority_accuracy_test = (test_data['sentiment'] == majority_label).mean()

print(f"Accuracy of the majority classifier on test_data: {round(majority_accuracy_test, 2)}")
