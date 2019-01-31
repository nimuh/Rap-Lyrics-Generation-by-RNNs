import pickle
import gc
import keras
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, BatchNormalization, Input, Flatten

def categorical(targets, size):
    """
    Takes the target values and converts to one-hot encodings.

    # Arguments:
        - targets: Target output data.
        - size: The number of unique characters.
    # Returns:
        - numpy array of one-hot encodings of target data.
    """
    categorized = []
    for target in targets:
        song_outputs = to_categorical(target, num_classes=size)
        categorized.append(song_outputs)
    return np.asarray(categorized)


def convert_word2word(lyrics, model_level):
    """
    Takes the scraped lyrics and converts it into input and target
    data.

    # Arguments:
        - lyrics: .txt file containing all lyrics
        - model_level: defines the model level. If 'char' then the
                       the input and targets generated are character
                       sequences. If 'word' then the input and targets
                       generated are sequences of words.
    # Returns:
        - inputs: The input data for the LSTM
        - outputs: The corresponding outputs for each input for the LSTM
        - class_size: The number of unique characters/words in lyrics
        - tokenizer.word_index: The dictionary of all characters/words and
                                their corresponding index.
        - window_size: The length of the input sequences. This is used to
                       define the input shape for the model.
    """
    # read scraped lyrics and split into lines
    file = open(lyrics, 'r')
    text = file.read()
    window_size = 5
    if model_level == 'char':
        window_size = 30
        doc = list(text)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(doc)
        tokenizer.word_index[' '] = 40
        tokenizer.word_index['\n'] = 41
        lyrics_as_index = [tokenizer.word_index[c] for c in doc]
    else:
        doc = text.split('\n')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(doc)
        input_data = text_to_word_sequence(text)
        lyrics_as_index = [tokenizer.word_index[word] for word in input_data]

    # use tokenizer to get integer representation of words of song
    inputs = []
    outputs = []

    for i in range(len(lyrics_as_index)-window_size):
        inputs.append(lyrics_as_index[i:i+window_size])
        outputs.append(lyrics_as_index[i+window_size])

    class_size = len(tokenizer.word_index)+1
    outputs = categorical(outputs, class_size)
    #print("OUTPUTS")
    #print(outputs.shape)
    inputs = np.asarray(inputs)
    inputs = inputs.reshape(inputs.shape[0], window_size, 1)

    return inputs, outputs, class_size, tokenizer.word_index, window_size


def data_gen(x, y, batch_size, vocab_size):
    """
    Creates batches of data on the fly.

    # Arguments:
        - x: inputs
        - y: outputs
        - batch_size: Size of batches for input
        - vocab_size: Number of unique members
    # Returns:
        - Data generator for creating batches.
    """
    nu_samples = x.shape[0]

    while True:

        samples = np.random.randint(low=0, high=nu_samples, size=(batch_size,))
        x_train = x[samples]
        y_train = y[samples]

        yield {'input': x_train}, {'output': y_train}

def recurrent_nn(vocab_size, window_size):

    """
    # Arguments:
        - vocab_size: Number of unique characters. Used as # of output units.
        - window_size: input sequence length
    # Returns:
        - A sequential Keras model with the following architecture:
            recurrent_nn defines the network architecture:
            Dense Layer: 100, ReLU
            Dense Layer: 300, ReLU
            Dense Layer: 30,  ReLU
            LSTM Layer:  256
            LSTM Layer:  256
            LSTM Layer:  256
            LSTM Layer:  256
            Dense Layer: vocab_size, softmax
    """

    input = Input(shape=(window_size, 1), name='input')
    dense1 = Dense(100, activation='relu')(input)
    batchNorm1 = BatchNormalization()(dense1)
    dense2 = Dense(300, activation='relu')(batchNorm1)
    batchNorm2 = BatchNormalization()(dense2)
    dense3 = Dense(30, activation='relu')(batchNorm2)
    batchNorm3 = BatchNormalization()(dense3)

    lstm1 = LSTM(256,
                 use_bias=True,
                 unit_forget_bias=True,
                 bias_initializer='ones',
                 #recurrent_dropout=0.2,
                 return_sequences=True)(batchNorm3)
    batchNorm4 = BatchNormalization()(lstm1)
    lstm2 = LSTM(256,
                 use_bias=True,
                 unit_forget_bias=True,
                 bias_initializer='ones',
                 #recurrent_dropout=0.2,
                 return_sequences=True)(batchNorm4)
    batchNorm5 = BatchNormalization()(lstm2)
    lstm3 = LSTM(256,
                 use_bias=True,
                 unit_forget_bias=True,
                 bias_initializer='ones',
                 #recurrent_dropout=0.2,
                 return_sequences=True)(batchNorm5)
    batchNorm6 = BatchNormalization()(lstm3)
    lstm4 = LSTM(256,
                 use_bias=True,
                 unit_forget_bias=True,
                 bias_initializer='ones',
                 #recurrent_dropout=0.2,
                 return_sequences=False)(batchNorm6)
    batchNorm7 = BatchNormalization()(lstm4)
    output = Dense(vocab_size, activation='softmax', name='output')(batchNorm7)

    model = Model(inputs=[input], outputs=[output])
    print(model.summary())
    return model


def train_model(nn, gen_tr, epochs):
    """
    Compiles and trains the model nn on X and y. Uses Adam as optimizer. All
    training history (accuracy, loss, etc) stored in history.pckl.

    # Arguments:
        - nn:     keras neural network model
        - gen_tr: data generator for creating batches of training data
        - epochs: number of training iterations over X
    """
    adam = optimizers.Adam()
    nn.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    checkpoints = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                  monitor='val_loss',
                                  period=50)
    history = nn.fit_generator(generator=gen_tr,
                               steps_per_epoch=100,
                               epochs=epochs,
                               callbacks=[])

    f = open('history.pckl', 'wb')
    pickle.dump(history.history, f)
    f.close()
    nn.save('trained_model.h5')
    del nn


"""
#################################### TRAINING #################################
"""
X, y, class_size, word_dict, window_size = convert_word2word('lyrics.txt',
                                                             model_level='char')
X = np.asarray(X, dtype=object)
rnn = recurrent_nn(class_size, window_size)
batch_size = 800
epochs = 100
gen_train = data_gen(X, y, batch_size, class_size)
train_model(rnn, gen_train, epochs)
