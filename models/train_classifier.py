"""
Classifier Trainer
Project: Disaster Response Pipeline

Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl

Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""

# imports
import sys
import os
import re
import time
import warnings
import pickle

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from sklearn.metrics import f1_score, make_scorer, classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from length_transformer import LengthTransformer

warnings.filterwarnings('ignore')

def load_data(database_filepath):
    """
    Loads data from SQL Database

    Args:
    database_filepath: SQL database file

    Returns:
    X pandas_dataframe: Features dataframe
    Y pandas_dataframe: Target dataframe
    category_names list: Target labels
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "Table"
    df = pd.read_sql_table(table_name,engine)

    # Drop rows where related is equal to 2
    df = df[df['related'] != 2]

    # "Child alone class" contains only only 1 possible value which makes useless predicting of such class.
    df = df.drop('child_alone', axis = 1)

    X,Y = df['message'], df.drop(['message', 'id', 'original', 'genre'], axis=1)

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text data

    Args:
    text str: Messages as text data

    Returns:
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """

    '''Para aumentar a acurácia da lemmatização foi construída uma função
    para mapear a POS Tag das palavras no texto, uma vez que as palavras
    sendo utilizadas como verbos, adjuntos, nomes, adverbios podem mudar
    a forma como a lemmatização é realizada.'''

    def get_wordnet_pos(word):
        """
        Tokenizes text data

        Args:
        text str: Messages as text data

        Returns:
        words list: Processed text after normalizing, tokenizing and lemmatizing
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)


    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)) for w in tokens]

    clean_tokens = []
    for tok in lemmed:
        clean_tok = tok.lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build model with GridSearchCV

    Returns:
    Trained model after performing grid search
    """
    # model pipeline
    pipeline = Pipeline([
                        ('features', FeatureUnion([

                            ('text_pipeline', Pipeline([
                                ('vect', CountVectorizer(tokenizer=tokenize, min_df=2, ngram_range=(1,2), max_df=0.8)),
                                ('tfidf', TfidfTransformer())])),

                            ('length_pipeline', Pipeline([
                                ('get_length', LengthTransformer())])),

                        ])),

                        ('clf', MultiOutputClassifier(estimator=LogisticRegression()))
                      ])

    # parameter grid
    parameters = {
        'clf__estimator__max_iter': [1500, 2000],
        'clf__estimator__C': [10],
        'clf__estimator__penalty': ['l2'],
        'clf__estimator__class_weight': ['balanced'],
        'clf__estimator__random_state': [42],
        'clf__estimator__n_jobs': [-1]
    }

    # create model
    f1_micro = make_scorer(f1_score, average='micro')
    cv_model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, scoring=f1_micro, verbose=3)

    return cv_model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Shows model's performance on test data

    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """
    y_pred = model.predict(X_test)

    overall_accuracy = (y_pred == Y_test).mean().mean()

    # Print the whole classification report
    Y_pred = pd.DataFrame(y_pred, columns = category_names)


    print(classification_report(Y_test, y_pred, target_names=category_names))

    i=0
    list_aux=[]
    for column in category_names:
        list_aux.append(balanced_accuracy_score(Y_test[column],Y_pred[column]))
        i = i + 1

    print('\nAverage overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    a = sum(list_aux)/len(list_aux)
    print('\nAverage balanced accuracy {0:.2f}%\n'.format((a)*100))


def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file
    Args:
    model: Trained model
    model_filepath: Filepath to save the model
    """

    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Train Classifier Main function

    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle='True', random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
