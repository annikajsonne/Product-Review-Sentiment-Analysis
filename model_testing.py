from scipy.sparse import vstack
import numpy as np

def predict_test(test_data, sentiment_model):
    # make predictions on sample test data
    sample_test_data = test_data.iloc[90:98]
    scores = []

    X_sample_test = vstack(sample_test_data["word_count_vec"].values)

    scores = sentiment_model.decision_function(X_sample_test)

    for i, score in enumerate(scores):
        print(f"\nReview {i+1}:")
        print("Rating:", sample_test_data.iloc[i]["rating"])
        print("Review:\n", sample_test_data.iloc[i]["review"])
        print("Score (z):", score)

    true_sentiment = "+1" if sample_test_data.iloc[i]["rating"] >= 4 else "-1"
    predicted_sentiment = "+1" if score > 0 else "-1"
    print("True Sentiment:", true_sentiment)
    print("Predicted Sentiment:", predicted_sentiment)

    compatible_scores_sample_test = sum(
        (score > 0 and sample_test_data.iloc[i]["rating"] >= 4)
        or (score <= 0 and sample_test_data.iloc[i]["rating"] <= 2)
        for i, score in enumerate(scores)
    )

    print(
        f"\nNumber of scores compatible with the true sentiment on the sample test data: {compatible_scores_sample_test}"
    )
    return sample_test_data, scores

def compute_class_predictions(scores) :
    # compute the class predictions (y hat)
    y_hat_sample_test = []

    for score in scores:
        if score > 0:
            y_hat_sample_test.append(+1)
        else:
            y_hat_sample_test.append(-1)

    print("Class predictions (y_hat):")
    print(y_hat_sample_test)
    return y_hat_sample_test

def compute_pos_proba(scores):
    # compute the probability that a sentiment is positive for each sample
    probabilities = 1 / (1 + np.exp(-scores))
    print("Probabilities of positive sentiment on the sample test data:")
    print(probabilities)
    return probabilities

# compute the accuracy of the classifier
def get_classification_accuracy(model, data, true_labels):
    """
    Compute the accuracy of the model on the given data and labels.
    """
    predictions = model.predict(data)
    correct_predictions = sum(predictions == true_labels)
    accuracy = correct_predictions / float(len(true_labels))
    return accuracy

def compare_classification_accuracies( sentiment_model, simple_model, X_train, y_train, X_train_subset, X_test, y_test, test_data):
    # compute the classification accuracies for both models on the train data
    training_accuracy_sentiment_model = get_classification_accuracy(
        sentiment_model, X_train, y_train
    )

    training_accuracy_simple_model = get_classification_accuracy(
        simple_model, X_train_subset, y_train
    )

    print(f"Training Accuracy of sentiment_model: {training_accuracy_sentiment_model}")
    print(f"Training Accuracy of simple_model: {training_accuracy_simple_model}")

    higher_accuracy_model = (
        "sentiment_model"
        if training_accuracy_sentiment_model > training_accuracy_simple_model
        else "simple_model"
    )
    print(f"Model with higher accuracy on the training set: {higher_accuracy_model}")

    # compute the classification accuracies for both models on the test data
    X_test_simple = vstack(test_data["word_count_subset_vec"].values)
    test_accuracy_sentiment_model = get_classification_accuracy(
        sentiment_model, X_test, y_test
    )
    test_accuracy_simple_model = get_classification_accuracy(
        simple_model, X_test_simple, y_test
    )
    print(f"Test Accuracy of sentiment_model: {test_accuracy_sentiment_model}")
    print(f"Test Accuracy of simple_model: {test_accuracy_simple_model}")

    higher_accuracy_model_test = (
        "sentiment_model"
        if test_accuracy_sentiment_model > test_accuracy_simple_model
        else "simple_model"
    )
    print(
        f"Model with higher accuracy on the testing set: {higher_accuracy_model_test}"
    )

def compare_pos_words(word_coef, cv, sentiment_model):
# determine whether the positive words in the simple model are also positive in the sentiment model
    positive_significant_words = [
        word for word, coef in word_coef.items() if coef > 0 and word != "intercept"
    ]
    sentiment_feature_names = cv.get_feature_names_out()
    sentiment_coefficients = dict(
        zip(sentiment_feature_names, sentiment_model.coef_[0])
    )
    all_positive_in_sentiment_model = all(
        sentiment_coefficients[word] > 0 for word in positive_significant_words
    )
    print(
        "Are all positive significant words from simple_model also positive in sentiment_model?",
        all_positive_in_sentiment_model,
    )