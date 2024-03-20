# Product Review Sentiment Analysis

This project implements a sentiment analysis system for product reviews using machine learning techniques. It aims to predict the sentiment (positive or negative) of product reviews based on the text content of the reviews.

## Overview

The project consists of several Python scripts:

- `main.py`: The main script that orchestrates the data processing, model training, testing, and evaluation.
- `data_processing.py`: Contains functions for loading and preprocessing the data.
- `model_training.py`: Includes functions for training logistic regression models.
- `model_testing.py`: Contains functions for testing the trained models and evaluating their performance.
- `utils.py`: Contains utility functions used throughout the project.

## Requirements
- Python 3.0
- Pip3

## Usage

1. Clone the repository:

```bash
git clone <repository_url>
cd Product-Review-Sentiment-Analysis
```

2. Run the main script
```
python3 main.py
```

## Scripts Overview

`main.py`: The main.py script is the entry point of the project. It imports necessary libraries, orchestrates the data processing, model training, testing, and evaluation, and prints out the results.

`data_processing.py`: The data_processing.py script contains functions for loading and preprocessing the data. It handles tasks such as loading data from CSV files, removing punctuation, and creating feature vectors using CountVectorizer.

`model_training.py`: The model_training.py script includes functions for training logistic regression models. It provides functions for training models on the full feature set and on a subset of significant words.

`model_testing.py`: The model_testing.py script contains functions for testing the trained models and evaluating their performance. It includes functions for predicting test data, computing class predictions, computing positive probabilities, and comparing classification accuracies.

`utils.py`: The utils.py script contains utility functions used throughout the project. It includes functions for feature engineering, extracting weights, computing the majority classifier, finding the most positive and negative words, and computing confusion matrices.

