from keras.models import Model
from keras import layers
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable, name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1), as_dict=True, signature='default',)['elmo']
        return result
    '''
    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')
    '''
    def compute_output_shape(self, input_shape):
        return (input_shape[0], None, self.dimensions)


def create_elmo_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen, drop_rate):
    input_text = layers.Input(shape=(None,), dtype=tf.string)
    embedding = ElmoEmbeddingLayer()(input_text)

    x = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv1D(num_filters, kernel_size)(embedding)))
    #x = layers.Dropout(drop_rate)(layers.MaxPooling1D()(x))

    #x = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv1D(num_filters, kernel_size)(x)))
    x = layers.Dropout(drop_rate)(layers.GlobalMaxPool1D()(x))

    x = layers.Dropout(drop_rate)(layers.Dense(num_filters, activation='relu')(x))
    #x =layers.Dropout(drop_rate)(layers.Dense(num_filters, activation='relu')(embedding))
    out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(input_text, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model



def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen, drop_rate):
    seq_input = layers.Input(shape=(maxlen,), dtype='float32')
    embedded_seq = layers.Embedding(vocab_size, embedding_dim, input_length=maxlen)(seq_input)

    #x = layers.BatchNormalization()(layers.Activation('relu')(layers.Conv1D(num_filters, kernel_size)(embedded_seq)))
    x = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv1D(num_filters, kernel_size)(embedded_seq)))
    x = layers.Dropout(drop_rate)(layers.MaxPooling1D(pool_size=3)(x))
    #x = layers.MaxPooling1D()(x)

    #x = layers.BatchNormalization()(layers.Activation('relu')(layers.Conv1D(num_filters, kernel_size)(embedded_seq)))
    #x = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv1D(num_filters, kernel_size)(x)))
    #x = layers.Dropout(drop_rate)(layers.MaxPooling1D()(x))
    #x = layers.MaxPooling1D()(x)

    #x = layers.BatchNormalization()(layers.Activation('relu')(layers.Conv1D(num_filters, kernel_size)(embedded_seq)))
    x = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv1D(num_filters, kernel_size)(x)))
    x = layers.Dropout(drop_rate)(layers.GlobalMaxPool1D()(x))
    #x = layers.GlobalMaxPool1D()(x)

    x = layers.Dropout(drop_rate)(layers.Dense(256, activation='relu')(x))
    out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(seq_input, out)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


