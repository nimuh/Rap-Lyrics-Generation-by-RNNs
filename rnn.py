import matplotlib.pyplot as plt
import pickle
import gc
import keras
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten, Input, Masking, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras import backend as K
from tensorflow.python.client import device_lib
gc.enable()

def categorical(targets, size):
    categorized = []
    for target in targets:
        song_outputs = to_categorical(target, num_classes=size)
        categorized.append(song_outputs)
    return np.asarray(categorized)

"""
convert_word2word takes the scraped lyrics and converts it into input and target
data for recurrent neural network.
"""
def convert_word2word(lyrics, model_level):

    # read scraped lyrics and split into lines
    file = open(lyrics, 'r')
    text = file.read()
    window_size = 5
    if model_level == 'char':
        window_size = 20
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
    inputs = np.asarray(inputs)
    inputs = inputs.reshape(inputs.shape[0], window_size, 1)

    return inputs, outputs, class_size, tokenizer.word_index, window_size


"""
recurrent_nn defines the network architecture:
1 Masking layer (skips padded areas due to variable sequence lengths)
1 LSTM layer
1 Dense layer (softmax output equal to number of unique words)
"""
def recurrent_nn(vocab_size, X, window_size):
    model = Sequential()
    model.add(Dense(100, input_shape=(window_size, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(20, activation='relu'))
    model.add(BatchNormalization())
    model.add(LSTM(256, use_bias=True, unit_forget_bias=True,
                                       bias_initializer='ones',
                                       recurrent_dropout=0.2,
                                       return_sequences=True))
    model.add(LSTM(256, use_bias=True, unit_forget_bias=True,
                                       bias_initializer='ones',
                                       recurrent_dropout=0.2,
                                       return_sequences=True))
    model.add(LSTM(256, use_bias=True, unit_forget_bias=True,
                                       bias_initializer='ones',
                                       recurrent_dropout=0.2,
                                       return_sequences=True))
    model.add(LSTM(256, use_bias=True, unit_forget_bias=True,
                                       bias_initializer='ones',
                                       recurrent_dropout=0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    return model

"""
train_model compiles and trains the rnn model obtained from recurrent_nn. Uses
RMSprop as optimizer.
"""
def train_model(nn, X, y, batch_size, epochs, val):
    adam = optimizers.Adam()
    nn.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    checkpoints = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                  monitor='val_loss',
                                  period=50)
    history = nn.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoints], validation_split=val)
    
    f = open('history.pckl', 'wb')
    pickle.dump(history.history, f)
    f.close()
    nn.save('trained_model.h5')
    del nn

def generate(word_values, network, length, model_level):
    keys = list(word_values.keys())
    number_to_word = list(word_values.values())
    rap = []
    if model_level == 'char':
        rap = ['t', 'o', 'o', ' ', 'l', 'a', 't', 'e', ' ', 'f']
    else:
        rap = ['too', 'late', 'for', 'me']
    current_length = window_size
    while current_length < length:
        current_rap = [word_values.get(key) for key in rap]
        current_rap = current_rap[current_length-window_size:current_length]
        pred = network.predict(np.asarray(current_rap).reshape(1, window_size, 1))
        pred = pred.astype(float)
        pred /= pred.sum()
        word_number = np.argmax(pred[0])
        word_number = np.argmax(np.random.multinomial(1, pred[0], size=1))
        rap.append(keys[number_to_word.index(word_number)])
        current_length += 1
    return rap

gc.collect()
X, y, class_size, word_dict, window_size = convert_word2word('lyrics.txt', model_level='char')
X = np.asarray(X, dtype=object)
rnn = recurrent_nn(class_size, X, window_size)

batch_size = 12000
epochs = 75
validation=0.50
train_model(rnn, X, y, batch_size, epochs, validation)

gc.collect()
