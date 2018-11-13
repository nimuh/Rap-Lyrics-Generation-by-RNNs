
import keras
import numpy as np
from keras import optimizers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Flatten, Input, Masking, Dropout
from keras.layers.embeddings import Embedding

def categorical(targets, size, padding_length):
    categorized = []
    for target in targets:
        song_outputs = to_categorical(target, num_classes=size)
        song_outputs = pad_sequences(song_outputs.T, maxlen=padding_length, padding='post')
        song_outputs = song_outputs.T
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
            songs.append(one_song[:len(one_song)-1])
            song_targets.append(one_song[1:len(one_song)])
            one_song = []
            #song_nu += 1
            #print("Song ", song_nu)
            #if song_nu == number_of_songs: break
        else:
            one_song.append(tokenized_lyrics[i])
    return songs, song_targets
"""
convert_word2word takes the scraped lyrics and converts it into input and target
data for recurrent neural network.
"""
def convert_word2word(lyrics):

    # read scraped lyrics and split into lines
    file = open(lyrics, 'r')
    text = file.read()
    #text = scrape_lyrics.clean_lyrics(text)
    doc = text.split('\n')

    # use tokenizer to get integer representation of words of song
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(doc)
    input_data, targets = split_by_song(text_to_word_sequence(text))
    inputs = []
    outputs = []

    max = 0
    for songs in input_data:
        if len(songs) > max: max = (len(songs)-1)

    for songs in input_data:
        song = [tokenizer.word_index[words] for words in songs]
        song = song[:len(song)-1]
        inputs.append(song)

    for songs in targets:
        song = [tokenizer.word_index[words] for words in songs]
        song = song[1:len(song)]
        outputs.append(song)

    class_size = len(tokenizer.word_index)+1
    outputs = categorical(outputs, class_size, max)
    inputs = pad_sequences(inputs, maxlen=max, padding='post')
    inputs = inputs.reshape(inputs.shape[0], max, 1)

    return inputs, outputs, class_size, tokenizer.word_index


"""
recurrent_nn defines the network architecture:
1 Masking layer (skips padded areas due to variable sequence lengths)
1 LSTM layer
1 Dense layer (softmax output equal to number of unique words)
"""
def recurrent_nn(vocab_size, X):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(None, 1)))
    model.add(LSTM(200, return_sequences=True))
    #model.add(Dropout(0.1))
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

def generate(word_values, network, length):
    keys = list(word_values.keys())
    number_to_word = list(word_values.values())
    rap = ['startss']
    current_length = 0
    while current_length < length:
        current_rap = [word_values.get(key) for key in rap]
        last_word = np.array(current_rap[-1]).reshape(1, 1, 1)
        pred = network.predict(last_word)[0][0]
        pred = pred.astype(float)
        pred /= pred.sum()
        word_number = np.argmax(np.random.multinomial(1, pred, size=1))+1
        while word_number == last_word:
            word_number = np.argmax(np.random.multinomial(1, pred, size=1))+1
        rap.append(keys[number_to_word.index(word_number)])
        current_length += 1
    return rap

X, y, class_size, word_dict = convert_word2word('lyrics.txt')
X = np.asarray(X, dtype=object)
rnn = recurrent_nn(class_size, X)

batch_size = 16
epochs = 175
validation=0.14
train_model(rnn, X, y, batch_size, epochs, validation)

print(generate(word_dict, rnn, 100))
