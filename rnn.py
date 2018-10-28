
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.layers.embeddings import Embedding

"""
convert takes the scraped lyrics and converts it into input and target
data for recurrent neural network.
"""
def convert(lyrics):

    # read scraped lyrics and split into lines
    file = open(lyrics, 'r')
    text = file.read()
    doc = text.split('\n')
    doc.insert(0, '[START]')
    doc.append('[END]')

    # use tokenizer to get integer representation of words of song
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(doc)
    word_seqs = text_to_word_sequence(text)

    input_data = tokenizer.texts_to_sequences(word_seqs)
    targets = input_data[1:len(input_data)]
    input_data = input_data[:len(input_data)-1]

    class_size = len(tokenizer.word_index)+1

    input_data = np.array(input_data)
    targets = np.array(targets)
    assert input_data.shape[0] == targets.shape[0]
    return input_data, targets, class_size

"""
recurrent_nn defines the network architecture:
1 Embedding layer
2 LSTM layers
2 Dense layers
"""
def recurrent_nn(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 315, input_length=seq_length))
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    return model

"""
train_model compiles and trains the rnn model obtained from recurrent_nn
"""
def train_model(nn, X, y, batch_size, epochs):
    nn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    nn.fit(X, y, batch_size=batch_size, epochs=epochs)


X, y, class_size= convert('lyrics.txt')
y = to_categorical(y, num_classes=class_size)
rnn = recurrent_nn(class_size, 1)

batch_size = 32
epochs = 500
train_model(rnn, X, y, batch_size, epochs)
