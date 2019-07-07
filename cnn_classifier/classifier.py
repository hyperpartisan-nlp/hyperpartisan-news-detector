import re
import pandas as pd
import string
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

from preprocessing import *
from model import *

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--elmo', help="use elmo embeddings", action='store_true', default=False)
#parser.add_argument('--genelmo', help="generate elmo embeddings", action='store_true', default=False)
#parser.add_argument('--seed', help="set seed for train-test-split", type=int, default=42)
args = argparser.parse_args()

seed=42
byart_tuple = ("../data/articles-training-byarticle-20181122.xml", "../data/ground-truth-training-byarticle-20181122.xml")
#bypub_tuple = ("../data/bypub-short.xml", "../data/ground-truth-training-bypublisher-20181122.xml")
#articles = parsemix_bp_ba_to_df(byart_tuple, bypub_tuple, cutoff=1000)
articles = parse_to_df(byart_tuple[0], byart_tuple[1])

X = articles['content']
ylabels = articles['hyperpartisan']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=seed)



bow_vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1,1))
#---------------------------------
#LogisticRegressionCV

classifier = LogisticRegressionCV(cv=4)
baseline = Pipeline([("cleaner", transformer()),("vectorizer", bow_vectorizer), ("classifier", classifier)])
baseline.fit(X_train, y_train)
prediction = baseline.predict(X_test)
print("LogisticRegression:")
print("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print("Precision: ", metrics.precision_score(y_test, prediction))
print("Recall: ", metrics.recall_score(y_test, prediction))
print("F1 Score: ", metrics.f1_score(y_test, prediction))

#---------------------------------
#SGDClassifier

sgd_clf = SGDClassifier(l1_ratio=0.15)
sgd_base = Pipeline([("cleaner", transformer()),("vectorizer", bow_vectorizer), ("classifier", sgd_clf)])
sgd_base.fit(X_train, y_train)
prediction = sgd_base.predict(X_test)
print("SGDClassifier:")
print("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print("Precision: ", metrics.precision_score(y_test, prediction))
print("Recall: ", metrics.recall_score(y_test, prediction))
print("F1 Score: ", metrics.f1_score(y_test, prediction))
#-----------------------------------------


X_train = X_train.apply(normalize, lowercase=True, remove_stopwords=True)
X_test = X_test.apply(normalize, lowercase=True, remove_stopwords=True)


if not args.elmo:
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

maxlen = 100
epochs = 20
if not args.elmo:
    vocab_size = len(tokenizer.word_index) + 1
else:
    vocab_size = 1

if not args.elmo:
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Param Grid
param_grid = dict(num_filters=[128],
        kernel_size=[5],
        vocab_size=[vocab_size],
        embedding_dim=[100],
        maxlen=[maxlen],
        drop_rate=[0.3])

if args.elmo:
    model = KerasClassifier(build_fn=create_elmo_model, epochs=epochs, batch_size=10, verbose=False)
else:
    model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=10, verbose=False)

#-------------------------
# RandomizedSearh

rando = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=1, n_jobs=1)
rando_result = rando.fit(X_train, y_train)

prediction_r = rando.predict(X_test)
print("CNN Classifier:")
print("Accuracy: ", metrics.accuracy_score(y_test, prediction_r))
print("Precision: ", metrics.precision_score(y_test, prediction_r))
print("Recall: ", metrics.recall_score(y_test, prediction_r))
print("F1 Score: ", metrics.f1_score(y_test, prediction_r))
#print("Best params: ", rando.best_params_)

