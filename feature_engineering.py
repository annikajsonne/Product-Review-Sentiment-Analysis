# feature_engineering.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import numpy as np

def create_subset_features(products, significant_words, train_data, test_data):
    """
    Create subset features using significant words.
    
    Parameters:
        products (pandas.DataFrame): Input DataFrame.
        significant_words (list): List of significant words.
        
    Returns:
        scipy.sparse.csr_matrix: Subset feature matrix.
        sklearn.feature_extraction.text.CountVectorizer: CountVectorizer instance.
    """
    # create a new CountVectorizer with only the significant words
    small_vectorizer = CountVectorizer(vocabulary=significant_words)

    subset_word_count_vec = small_vectorizer.fit_transform(
        products["review"].fillna("")
    )

    products["word_count_subset_vec"] = list(subset_word_count_vec)

        # add the new column to the training and testing dataframe
    train_data["word_count_subset_vec"] = products["word_count_subset_vec"]
    test_data["word_count_subset_vec"] = products["word_count_subset_vec"]

    return small_vectorizer

def extract_weights(cv, sentiment_model):
    # extract the weights of the words and store them in a dictionary that maps feature names to coefficients
    word_coef = {}
    feature_names = cv.get_feature_names_out()
    coefficients = sentiment_model.coef_[0]
    word_coef = dict(zip(feature_names, coefficients))

def extract_simple_weights(small_vectorizer, simple_model) :
      # extract the weights of the words and store them in a dictionary that maps feature names to coefficients
    feature_names_small_vec = small_vectorizer.get_feature_names_out()
    coefficients_small_vec = simple_model.coef_[0]
    intercept_small_vec = simple_model.intercept_[0]
    word_coef = dict(zip(feature_names_small_vec, coefficients_small_vec))
    word_coef["intercept"] = intercept_small_vec
    for word, coef in sorted(word_coef.items(), key=lambda item: item[1], reverse=True):
        print(f"{word}: {coef}")
    return word_coef

# function to return a majority classifier for the column label of frame data
def compute_majority_classifier(data, label):
    majority_label = data[label].mode()[0]

    def majority_classifier(_):
        return majority_label

    return majority_classifier

def engineer_majority_classifier(train_data, test_data):
    # compute the majority classifier
    majority_classifier_train = compute_majority_classifier(train_data, "sentiment")

    majority_class = majority_classifier_train(None)
    print(f"The majority classifier always predicts: {majority_class}")

    # compute the accuracy of the majority classifier on the test data
    majority_label = train_data["sentiment"].mode()[0]

    majority_accuracy_test = (test_data["sentiment"] == majority_label).mean()

    print(
        f"Accuracy of the majority classifier on test_data: {round(majority_accuracy_test, 2)}"
    )

def find_10_pos_neg(cv, sentiment_model):
    # find 10 most positive and 10 most negative weights in the learned model
    feature_names = cv.get_feature_names_out()
    coefficients = sentiment_model.coef_[0]
    features_coefficients = list(zip(feature_names, coefficients))
    most_positive_features = sorted(
        features_coefficients, key=lambda x: x[1], reverse=True
    )[:10]
    most_negative_features = sorted(features_coefficients, key=lambda x: x[1])[:10]
    print("10 Most Positive Weights:")
    for feature, coef in most_positive_features:
        print(f"{feature}: {coef}")
    print("\n10 Most Negative Weights:")
    for feature, coef in most_negative_features:
        print(f"{feature}: {coef}")

def find_min_pos_review(probabilities, sample_test_data):
        # identify the review classified as positive with the lowest probability and check if its prediction was correct
    min_positive_prob = min(prob for prob in probabilities if prob > 0.5)
    lowest_prob_index = np.argmin([prob if prob > 0.5 else 1 for prob in probabilities])
    print(
        f"Review classified as positive with the lowest probability in sample test data ({min_positive_prob}):"
    )
    print(sample_test_data.iloc[lowest_prob_index]["review"])
    print("Rating:", sample_test_data.iloc[lowest_prob_index]["rating"])

def compute_confusion_matrix(sentiment_model, X_test, y_test):
    y_pred = sentiment_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    return tp, tn, fp, fn