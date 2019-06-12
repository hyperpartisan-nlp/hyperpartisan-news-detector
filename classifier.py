import xml.etree.ElementTree as ET
import re
from html2text import HTML2Text
import spacy
import pandas as pd
from spacy.lang.en import English
import string

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


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
        article['content'] = text
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





x_articles = parse_articles("../articles-training-byarticle-20181122.xml")
y_articles = parse_hyperpartisan("../ground-truth-training-byarticle-20181122.xml")
articles = join(x_articles, y_articles)
articles = pd.DataFrame(articles)

parser = English()
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

bow_vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1,1))
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer)
classifier = LogisticRegression()

X = articles['content']
ylabels = articles['hyperpartisan']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

pipeline = Pipeline([("cleaner", transformer()),("vectorizer", bow_vectorizer), ("classifier", classifier)])
pipeline.fit(X_train, y_train)
print("Done fitting!")

prediction = pipeline.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print("Precision: ", metrics.precision_score(y_test, prediction))
print("Recall: ", metrics.recall_score(y_test, prediction))
print("F1 Score: ", metrics.f1_score(y_test, prediction))
