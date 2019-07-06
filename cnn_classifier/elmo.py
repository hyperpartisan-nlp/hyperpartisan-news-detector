import tensorflow as tf
import tensorflow_hub as hub
import pickle
from preprocessing import *


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable, name="{}_module".format(self.name))

def elmo_vectors(x):
    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1)

# Parse to pandas DataFrame (content, hyperpartisan)
articles = parse_to_df("../data/articles-training-byarticle-20181122.xml", "../data/ground-truth-training-byarticle-20181122.xml")

articles['content'] = articles['content'].apply(normalize, lowercase=False, remove_stopwords=True)

X = articles['content']
ylabels = articles['hyperpartisan']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25)

X_train = X_train.apply(normalize, lowercase=False, remove_stopwords=True)
X_test = X_test.apply(normalize, lowercase=False, remove_stopwords=True)

list_train = [X_train[i:i+50] for i in range(0,X_train.shape[0],50)]
list_test = [X_test[i:i+50] for i in range(0,X_test.shape[0],50)]

elmo_train = [elmo_vectors(x['content']) for x in list_train]
elmo_test = [elmo_vectors(x['content']) for x in list_test]


elmo_train_new = np.concatenate(elmo_train, axis = 0)
elmo_test_new = np.concatenate(elmo_test, axis = 0)

# save elmo_train_new
pickle_out = open("elmo_train_03032019.pickle","wb")
pickle.dump(elmo_train_new, pickle_out)
pickle_out.close()

# save elmo_test_new
pickle_out = open("elmo_test_03032019.pickle","wb")
pickle.dump(elmo_test_new, pickle_out)
pickle_out.close()


# load elmo_train_new
pickle_in = open("elmo_train_03032019.pickle", "rb")
elmo_train_new = pickle.load(pickle_in)

# load elmo_train_new
pickle_in = open("elmo_test_03032019.pickle", "rb")
elmo_test_new = pickle.load(pickle_in)
