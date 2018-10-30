
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten, Input
from keras.layers.embeddings import Embedding

"""
convert_word2word takes the scraped lyrics and converts it into input and target
data for recurrent neural network.
"""
def convert_word2word(lyrics):

    # read scraped lyrics and split into lines
    file = open(lyrics, 'r')
    text = file.read()
    doc = text.split('\n')
    doc.insert(0, '[START]')
    doc.append('[END]')

    # use tokenizer to get integer representation of words of song
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(doc)
    input_data = text_to_word_sequence(text)
    targets = input_data[1:len(input_data)]
    targets = [tokenizer.word_index[i] for i in targets]
    input_data = input_data[:len(input_data)-1]
    input_data = [tokenizer.word_index[i] for i in input_data]
    class_size = len(tokenizer.word_index)+1
    input_data = np.array(input_data)
    targets = np.array(targets)
    input_data = input_data.reshape((1, input_data.shape[0], 1))
    targets = targets.reshape((1, targets.shape[0], 1))

    print(input_data[0])
    print(targets[0])

    assert input_data.shape[0] == targets.shape[0]
    return input_data, targets, class_size


"""
recurrent_nn defines the network architecture:
1 LSTM layer
1 Dense layer
"""
def recurrent_nn(vocab_size, X):
    input_seq = Input(shape=(X.shape[1], 1))
    lstm1 = LSTM(200, return_sequences=True)(input_seq)
    output = Dense(vocab_size, activation='softmax')(lstm1)
    model = Model(input_seq, output)
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


X, y, class_size= convert_word2word('lyrics.txt')
y = to_categorical(y, num_classes=class_size)
rnn = recurrent_nn(class_size, X)

batch_size = 32
epochs = 300
train_model(rnn, X, y, batch_size, epochs)
