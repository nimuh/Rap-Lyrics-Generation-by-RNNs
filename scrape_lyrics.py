import lyricsgenius as lg
import string
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

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
    for seq in range(len(sequences)):
        if len(sequences[seq]) == 0:
            continue
        input_data.append(np.array(sequences[seq][:-1]))
        targets.append(np.array(sequences[seq][len(sequences[seq])-1]))
    class_size = len(tokenizer.word_index)+1

    input_data = np.array(input_data)
    targets = np.array(targets)
    assert input_data.shape[0] == targets.shape[0]

    return input_data, targets, class_size

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


search_by_artists('test.txt', 'out.txt', 1)
X, y, class_size = convert('out.txt')
print("INPUTS: ")
print(X)
print("TARGETS: ")
print(y)
