import lyricsgenius as lg
import string
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.layers.embeddings import Embedding
print(tf.__version__)  # for Python 3

key = "h7teesaCKXvcUvK8yxFIN1cSBSGchQ6i0MVlwE8YORNByy7N5U19geJqnKXxjCa1"
api = lg.Genius(key)

"""
clean_lyrics is used for removing tags from the lyrics.
"""
def clean_lyrics(doc):
    for word in doc:
        doc = doc.replace('[', '')
        doc = doc.replace(']', '')
        doc = doc.replace('(', '')
        doc = doc.replace(')', '')
        doc = doc.replace('Verse', '')
        doc = doc.replace('Chorus', '')
        doc = doc.replace('Outro', '')
        doc = doc.replace('Intro', '')
        doc = doc.replace('Hook', '')
        doc = doc.replace('Interlude', '')
    return doc

"""
convert takes the scraped lyrics and converts it into input and target
data for recurrent neural network.
"""
def convert(lyrics):

    # read scraped lyrics and split into lines
    file = open(lyrics, 'r')
    text = file.read()
    doc = text.split('\n')

    # use tokenizer to get integer representation of lines of song
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(doc)
    sequences = tokenizer.texts_to_sequences(doc) # <---|
    sequences = np.array(sequences)

    # split data into inputs and targets
    input_data = []
    targets = []
    max_length = 0
    for seq in range(len(sequences)):
        if len(sequences[seq]) == 0:
            continue
        if len(sequences[seq][:-1]) > max_length:
            max_length = len(sequences[seq][:-1])
        input_data.append(np.array(sequences[seq][:-1]))
        targets.append(np.array(sequences[seq][len(sequences[seq])-1]))
    class_size = len(tokenizer.word_index)+1

    # zero-pad input data so all inputs are of the same size
    input_data = np.array(pad_sequences(input_data,
                                        maxlen=max_length,
                                        padding='post'))
    targets = np.array(targets)
    assert input_data.shape[0] == targets.shape[0]
    return input_data, targets, max_length, class_size

"""
search_by_artists takes a list of rapper names and searches for
songs by these artists and scrapes the lyrics of songs found. All lyrics
are saved to one file.
"""
def search_by_artists(list_of_rappers, out_filename, nu_songs):
    with open(list_of_rappers) as f:
        file_content = f.readlines()
    file_content = [name.strip() for name in file_content]
    file = open(out_filename, 'w')
    for rapper in file_content:
        artist = api.search_artist(rapper, max_songs=nu_songs, get_full_song_info=False)
        for s in range(len(artist.songs)):
            doc = artist.songs[s].lyrics
            doc = clean_lyrics(doc)
            doc = doc.split('\n')
            for word in doc:
                file.write(word+'\n')
    file.close()

def recurrent_nn(X, y, vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=seq_length))
    #model.add(LSTM(100))
    #model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())


search_by_artists('test.txt', 'out.txt', 1)
X, y, max_length, class_size = convert('out.txt')
y = to_categorical(y, num_classes=class_size)
print(class_size)
recurrent_nn(X, y, class_size, max_length)
