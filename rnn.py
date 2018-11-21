
import keras
import numpy as np
from keras import optimizers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten, Input, Masking, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding

def categorical(targets, size, padding_length):
    categorized = []
    for target in targets:
        song_outputs = to_categorical(target, num_classes=size)
        categorized.append(song_outputs)
    return np.asarray(categorized)

def split_by_song(tokenized_lyrics):
    songs = []
    song_targets = []
    one_song = []
    one_song.append(tokenized_lyrics[0])
    song_nu = 0
    for i in range(1, len(tokenized_lyrics)):
        if tokenized_lyrics[i] == 'endss':
            one_song.append(tokenized_lyrics[i])
            songs.append(one_song[:len(one_song)]) # removed -1
            song_targets.append(one_song[1:len(one_song)])
            one_song = []
        else:
            one_song.append(tokenized_lyrics[i])
    return songs, song_targets
"""
convert_word2word takes the scraped lyrics and converts it into input and target
data for recurrent neural network.
"""
def convert_word2word(lyrics, window_size):

    # read scraped lyrics and split into lines
    file = open(lyrics, 'r')
    text = file.read()
    doc = text.split('\n')

    # use tokenizer to get integer representation of words of song
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(doc)
    input_data = text_to_word_sequence(text)
    inputs = []
    outputs = []

    lyrics_as_index = [tokenizer.word_index[words] for words in input_data]

    for i in range(len(lyrics_as_index)-window_size):
        inputs.append(lyrics_as_index[i:i+window_size])
        outputs.append(lyrics_as_index[i+window_size])

    class_size = len(tokenizer.word_index)+1
    outputs = categorical(outputs, class_size, max)
    inputs = np.asarray(inputs)
    inputs = inputs.reshape(inputs.shape[0], window_size, 1)

    return inputs, outputs, class_size, tokenizer.word_index


"""
recurrent_nn defines the network architecture:
1 Masking layer (skips padded areas due to variable sequence lengths)
1 LSTM layer
1 Dense layer (softmax output equal to number of unique words)
"""
def recurrent_nn(vocab_size, X, window_size):
    model = Sequential()
    model.add(Dense(200, input_shape=(window_size, 1), activation='relu'))
    model.add(LSTM(128, use_bias=True, unit_forget_bias=True, return_sequences=True))
    model.add(LSTM(128, use_bias=True, unit_forget_bias=True, return_sequences=True))
    model.add(LSTM(128, use_bias=True, unit_forget_bias=True))
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(400, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    return model

"""
train_model compiles and trains the rnn model obtained from recurrent_nn. Uses
RMSprop as optimizer.
"""
def train_model(nn, X, y, batch_size, epochs, val):
    rmsp = optimizers.RMSprop(lr=0.001)
    nn.compile(loss='categorical_crossentropy',
                  optimizer=rmsp,
                  metrics=['accuracy'])
    nn.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=val)
    nn.save('rap_lstm.h5')

def generate(word_values, network, length, window_size):
    keys = list(word_values.keys())
    number_to_word = list(word_values.values())
    rap = ['startss', 'too', 'late', 'for', 'me']
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

window_size = 5
X, y, class_size, word_dict = convert_word2word('lyrics.txt', window_size)
X = np.asarray(X, dtype=object)
print(X.shape)
print(y.shape)
rnn = recurrent_nn(class_size, X, window_size)

batch_size = 256
epochs = 200
validation=0.0
train_model(rnn, X, y, batch_size, epochs, validation)

print(generate(word_dict, rnn, 200, window_size))
