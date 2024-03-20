# model_training.py
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(train_data):
    """
    Train a logistic regression model.
    
    Parameters:
        X_train (scipy.sparse.csr_matrix): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        C (float): Inverse of regularization strength.
        penalty (str): Regularization penalty ('l1' or 'l2').
        solver (str): Optimization algorithm.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        sklearn.linear_model.LogisticRegression: Trained model.
    """
    print('Training the model...')

    X_train = vstack(train_data['word_count_vec'].values)

    y_train = train_data['sentiment'].values

    sentiment_model = LogisticRegression(C=100, penalty='l2', solver='lbfgs', max_iter=1000)
    sentiment_model.fit(X_train, y_train)

    print('Model Trained.')

    return sentiment_model, X_train, y_train


def train_simple_model(train_data):
    """
    Train a logistic regression model on the subset of words.
    """
    print("Training the simple model...")

    # train a logistic regression model on the subset of words
    X_train_subset = vstack(train_data["word_count_subset_vec"].values)
    y_train = train_data["sentiment"].values

    simple_model = LogisticRegression(
        C=100, penalty="l2", solver="lbfgs", max_iter=1000
    )
    simple_model.fit(X_train_subset, y_train)

    print("Simple model trained.")
    return simple_model, X_train_subset, y_train