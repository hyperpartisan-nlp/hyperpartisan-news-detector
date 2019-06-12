# Author: Daan Middendorp <d.middendorp@campus.tu-berlin.de>

from time import time
import logging
import xml.etree.ElementTree as ET
import math

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

categories = [
    'hyperpartisan',
    'non-hyperpartisan',
]

articles = []
target_category = []

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(tol=1e-3)),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (20,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__max_iter': (10, 50, 80),
}

validation_percentage = 0.2

if __name__ == "__main__":

    tree = ET.parse('data/articles-training-byarticle-20181122.xml')
    for country in tree.findall('article'):
        articles.append(ET.tostring(country))

    truth = ET.parse('data/ground-truth-training-byarticle-20181122.xml')
    for country in truth.findall('article'):
        target_category.append(1 if country.get('hyperpartisan') == 'true' else 0)

    validation_len = math.floor(len(articles)*validation_percentage)
    training_len = len(articles)-validation_len

    train_articles = articles[:training_len]
    train_hyperpartisan = target_category[:training_len]
    validation_articles = articles[validation_len*-1:]
    validation_hyperpartisan = target_category[validation_len*-1:]

    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    t0 = time()
    grid_search.fit(train_articles, train_hyperpartisan)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


    prediction = grid_search.predict(validation_articles)

    correct = (prediction == validation_hyperpartisan)
    accuracy = correct.sum() / correct.size
    print(accuracy)