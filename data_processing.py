import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
        
    Returns:
        pandas.DataFrame: Loaded data.
    """
    products = pd.read_csv(file_path)
    print('There are',len(products),'reviews of baby products')
    return products

def remove_punctuation(text):
    """
    Remove punctuation from text.
    
    Parameters:
        text (str): Input text.
        
    Returns:
        str: Text with punctuation removed.
    """
    if not isinstance(text, str):
        # If not, return an empty string or some other placeholder
        return ""
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)

def preprocess_data(products):
    """
    Preprocess data by removing punctuation and handling missing values.
    
    Parameters:
        products (pandas.DataFrame): Input DataFrame.
        
    Returns:
        None
    """
    products['review'].fillna('', inplace=True)
    review_no_punctuation = products['review'].apply(remove_punctuation)    


def create_feature_vectors(products, stop_words, min_df=2, max_df=0.6):
    """
    Create feature vectors using CountVectorizer.
    
    Parameters:
        text_data (pandas.Series): Text data.
        stop_words (list): List of stop words.
        min_df (int): Minimum document frequency.
        max_df (float): Maximum document frequency.
        
    Returns:
        None
    """
    cv = CountVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df)
    word_count_vector = cv.fit_transform(products['review'])
    products['word_count_vec'] = [sparse.csr_matrix(word_count_vector[i]) for i in range(word_count_vector.shape[0])]
    return cv

