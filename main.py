from scipy.sparse import vstack
from sklearn.metrics import precision_score, recall_score

from data_processing import (
    load_data,
    preprocess_data,
    create_feature_vectors,
)
from model_training import train_model, train_simple_model
from utils import (
    create_subset_features,
    extract_weights,
    extract_simple_weights,
    engineer_majority_classifier,
    find_10_pos_neg,
    find_min_pos_review,
    compute_confusion_matrix,
)
from model_testing import (
    predict_test,
    compute_class_predictions,
    compute_pos_proba,
    compare_classification_accuracies,
    compare_pos_words,
    get_classification_accuracy
)

import warnings

warnings.filterwarnings("ignore")
# use WIDER CANVAS:
from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))


def main():
    # import data
    products = load_data("./data/amazon_baby.gz")

    # preprocess data
    preprocess_data(products)

    # create feature vectors
    stop_words = [
        "you",
        "he",
        "she",
        "they",
        "an",
        "the",
        "and",
        "or",
        "in",
        "on",
        "at",
        "is",
        "was",
        "were",
        "am",
        "are",
    ]
    cv = create_feature_vectors(products, stop_words)

    # create sentiment
    # ignore all neutral sentiment reviews, where rating is equal to 3
    products = products[products['rating'] != 3]
    # create a sentiment column and consider reviews with a rating of 4 or higher as +1 and a rating of 2 or lower as -1
    products['sentiment'] = products['rating'].apply(lambda x: 1 if x >= 4 else (-1 if x <= 2 else 0))

    # split the data
    train_data = products.sample(frac=0.8, random_state=0)
    test_data = products.drop(train_data.index)
    print(f"We will use N={len(train_data)} training samples")
    print(f"and {len(test_data)} testing samples")
    print(f"Total samples: {len(products)}")

    print(f'\n')
    # train the model
    sentiment_model, X_train, y_train = train_model(train_data)

    # extract the weights of the words and store them in a dictionary that maps feature names to coefficients
    extract_weights(cv, sentiment_model)
    print(f'\n')

    # make predictions on sample test data
    sample_test_data, scores = predict_test(test_data, sentiment_model)
    print(f'\n')

    # compute the class predictions
    compute_class_predictions(scores)
    print(f'\n')

    # compute the probability that a sentiment is positive for each sample
    probabilities = compute_pos_proba(scores)
    print(f'\n')

    # find the review with the minimum probability of being positive
    find_min_pos_review(probabilities, sample_test_data)
    print(f'\n')

    # compute the accuracy of the classifier
    X_test = vstack(test_data['word_count_vec'].values)
    y_test = test_data['sentiment'].values
    training_accuracy = get_classification_accuracy(sentiment_model, X_train, y_train)
    test_accuracy = get_classification_accuracy(sentiment_model, X_test, y_test)
    print(f"Training Accuracy: {round(training_accuracy, 4)}")
    print(f"Test Accuracy: {round(test_accuracy, 4)}")
    print(f'\n')

    # compare the most positive and negative words
    find_10_pos_neg(cv, sentiment_model)
    print(f'\n')

    # learn another classifier with fewer words
    significant_words = [
        "love",
        "great",
        "easy",
        "old",
        "little",
        "perfect",
        "loves",
        "wonderfully",
        "lifesaver",
        "well",
        "broke",
        "less",
        "waste",
        "disappointed",
        "unusable",
        "work",
        "money",
        "return",
    ]

    # create a new CountVectorizer with only the significant words
    small_vectorizer = create_subset_features(
        products, significant_words, train_data, test_data
    )
    print(f'\n')

    # train a logistic regression model on the subset of words
    simple_model, X_train_subset, y_train = train_simple_model(train_data)
    print(f'\n')
    # extract the weights of the words and store them in a dictionary that maps feature names to coefficients
    word_coef = extract_simple_weights(small_vectorizer, simple_model)
    compare_pos_words(word_coef, cv, sentiment_model)
    print(f'\n')

    # compare classification accuracy
    compare_classification_accuracies(
        sentiment_model,
        simple_model,
        X_train,
        y_train,
        X_train_subset,
        X_test,
        y_test,
        test_data,
    )
    print(f'\n')

    # compute the majority classifier
    engineer_majority_classifier(train_data, test_data)

    # exploring precision and recall
    # compute confusion matrix
    tp, tn, fp, fn = compute_confusion_matrix(sentiment_model, X_test, y_test)

    y_pred = sentiment_model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    # determine if the model has a recal >= 95%
    recall_for_positive_class = tp / (tp + fn)
    print(f"Recall: {recall:.3f}")
    meets_threshold = recall >= 0.95
    print(f"Does the model's recall meet the threshold of 95%? {'Yes' if meets_threshold else 'No'}")

    print(
        f"Fraction of positive reviews correctly predicted as positive: {recall_for_positive_class}"
    )
    print(f'\n')

    # assign costs to false positives and false negatives
    cost_false_negative = 100
    cost_false_positive = 1
    total_cost = (cost_false_negative * fn) + (cost_false_positive * fp)
    print(f"Total cost on the test data: ${total_cost}")


if __name__ == "__main__":
    main()
