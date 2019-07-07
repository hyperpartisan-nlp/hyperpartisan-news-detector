from html2text import HTML2Text
import xml.etree.ElementTree as ET
import spacy
import pandas as pd
from sklearn.base import TransformerMixin
from spacy.lang.en import English
import string

nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
parser = English()
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

def clean_content(content, h):
    ''' Clean article content.'''
    # remove anchor links, numbers, whitespace, stopwords
    content = h.handle(content)
    #content = content.lower().strip()
    content = ''.join([ch if (ch not in punctuations and not ch.isdigit()) else ' ' for ch in content])
    return content

def clean_articles(articles):
    """ Cleans articles by removing HTML tags. """
    h = HTML2Text()
    for article in articles:
        article['content'] = clean_content(article['content'], h)
    return articles

def parse_articles(filepath, counter=1000):
    ''' Parse article file and convert into mapping.'''
    tree = ET.parse(filepath)
    root = tree.getroot()

    articles = []
    for child in root:
        if counter == 0:
            break
        article = {'id' : child.attrib['id']}
        text = ""
        for text_node in child:
            if text_node.tag == 'p':
                if text_node.text is not None:
                    text += text_node.text
        article['content'] = text
        articles.append(article)
        counter -= 1
    return clean_articles(articles)

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

def parse_to_df(articles_path, labels_path):
    X = parse_articles(articles_path)
    y = parse_hyperpartisan(labels_path)
    return pd.DataFrame(join(X,y))

# pre-process texts to remove stop words
def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stop_words):
                lemmatized.append(lemma)
    return " ".join(lemmatized)

def parsemix_bp_ba_to_df(byart_tuple, bypub_tuple, cutoff=1000):
    byarticle = parse_to_df(byart_tuple[0], byart_tuple[1])
    bypublisher = parse_to_df(bypub_tuple[0], bypub_tuple[1])
    #bypublisher.drop(range(cutoff))
    data = byarticle.append(bypublisher)
    return data


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
        return [text.strip().lower() for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

