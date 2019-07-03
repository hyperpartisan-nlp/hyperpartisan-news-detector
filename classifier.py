import xml.etree.ElementTree as ET
import re
from html2text import HTML2Text
import spacy
import pandas as pd
from spacy.lang.en import English
import string

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import FunctionTransformer 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

from keras.models import Model
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt

import logging
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def parse_articles(filepath):
    ''' Parse article file and convert into mapping.'''
    tree = ET.parse(filepath)
    root = tree.getroot()

    articles = []
    for child in root:
        #print(child.attrib)
        article = {'id' : child.attrib['id']}
        text = ""
        for text_node in child:
            if text_node.tag == 'p':
                if text_node.text is not None:
                    text += text_node.text
        article['content'] = text + child.attrib['title']
        articles.append(article)
    return clean_articles(articles)

def clean_content(content, h):
    ''' Clean article content.'''
    # remove anchor links
    content = h.handle(content)
    content.strip().lower()
    #content = content.replace('\(', ' ')
    #content = re.sub("<(?:a\b[^>]*>|[/a>)", ' ', content)
    return content

def clean_articles(articles):
    """ Cleans articles by removing HTML tags. """
    h = HTML2Text()
    for article in articles:
        article['content'] = clean_content(article['content'], h)
    return articles

def tokenizer(sentence):
    """ Tokenizes a sentence. """
    # use English parser
    tokens = parser(sentence)
    # Lemmatization
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    # Remove stop words and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations]
    return tokens

# Custom transformer using spaCy
class transformer(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

def parse_hyperpartisan(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    articles = []
    for child in root:
        article = {"id" : child.attrib['id'], "hyperpartisan" : child.attrib['hyperpartisan']}
        articles.append(article)

    return articles

def join(x_articles, y_articles):
    joined = []
    for x in x_articles:
        for y in y_articles:
            if x['id'] == y['id']:
                joined.append({'id' : x['id'], 'hyperpartisan' : 1 if y['hyperpartisan'] == 'true' else 0, 'content' : x['content']})
    return joined
'''
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
'''

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):

    seq_input = layers.Input(shape=(maxlen,), dtype='float32')
    embedded_seq = layers.Embedding(vocab_size, embedding_dim, input_length=maxlen)(seq_input)

    x = layers.Conv1D(num_filters, kernel_size, activation='relu')(embedded_seq)
    x = layers.MaxPooling1D()(x)

    x = layers.Conv1D(num_filters, kernel_size, activation='relu')(x)
    x = layers.MaxPooling1D()(x)

    x = layers.Conv1D(num_filters, kernel_size, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)

    #x = layers.Flatten()(x)

    x = layers.Dense(num_filters, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(seq_input, out)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

x_articles = parse_articles("./data/articles-training-byarticle-20181122.xml")
y_articles = parse_hyperpartisan("./data/ground-truth-training-byarticle-20181122.xml")
articles = join(x_articles, y_articles)
articles = pd.DataFrame(articles)

parser = English()
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

X = articles['content']
ylabels = articles['hyperpartisan']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25)

hyper = []
nonhyper = []

for i in range(len(articles['content'])):
    if ylabels[i] == 1:
        hyper.append(len(articles['content'][i].split()))
    else:
        nonhyper.append(len(articles['content'][i].split()))

print(np.mean(hyper))
print(np.mean(nonhyper))


bow_vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1,1))

#---------------------------------
#LogisticRegression



def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)


classifier = LogisticRegressionCV(cv=5)

baseline = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vectorizer', bow_vectorizer),
            ('tfidf', TfidfTransformer()),
        ])),
        ('length', Pipeline([
            ('count', FunctionTransformer(get_text_length, validate=False)),
        ]))
    ])),
    ('clf', LogisticRegressionCV(cv=5))])

baseline.fit(X_train, y_train)
prediction = baseline.predict(X_test)
print("Logistic Regression")
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
print("SGD Classifier")
print("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print("Precision: ", metrics.precision_score(y_test, prediction))
print("Recall: ", metrics.recall_score(y_test, prediction))
print("F1 Score: ", metrics.f1_score(y_test, prediction))

#-----------------------------------------
#Random Forest Classifier

rf_clf = RandomForestClassifier()
rf_base = Pipeline([("cleaner", transformer()),("vectorizer", bow_vectorizer), ("classifier", rf_clf)])
rf_base.fit(X_train, y_train)
prediction = rf_base.predict(X_test)
print("Random Forest Classifier")
print("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print("Precision: ", metrics.precision_score(y_test, prediction))
print("Recall: ", metrics.recall_score(y_test, prediction))
print("F1 Score: ", metrics.f1_score(y_test, prediction))

#-----------------------------------------
#Naive Bayes Classifier

nb_clf = MultinomialNB()
nb_base = Pipeline([("cleaner", transformer()),("vectorizer", bow_vectorizer), ("classifier", nb_clf)])
nb_base.fit(X_train, y_train)
prediction = nb_base.predict(X_test)
print("Naive Base Classifier")
print("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print("Precision: ", metrics.precision_score(y_test, prediction))
print("Recall: ", metrics.recall_score(y_test, prediction))
print("F1 Score: ", metrics.f1_score(y_test, prediction))
#-----------------------------------------


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

epochs = 20
vocab_size = len(tokenizer.word_index) + 1
maxlen = 100
embedding_dim = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Param Grid
param_grid = dict(num_filters=[128],
        kernel_size=[7],
        vocab_size=[vocab_size],
        embedding_dim=[embedding_dim],
        maxlen=[maxlen])


model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=10, verbose=False)

#-------------------------
# RandomizedSearh

rando = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=1, n_jobs=1)
rando_result = rando.fit(X_train, y_train)

prediction_r = rando.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, prediction_r))
print("Precision: ", metrics.precision_score(y_test, prediction_r))
print("Recall: ", metrics.recall_score(y_test, prediction_r))
print("F1 Score: ", metrics.f1_score(y_test, prediction_r))
print("Best params: ", rando.best_params_)

