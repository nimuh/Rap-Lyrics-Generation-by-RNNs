import lyricsgenius as lg
import string
import keras
import word2vec
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


key = "h7teesaCKXvcUvK8yxFIN1cSBSGchQ6i0MVlwE8YORNByy7N5U19geJqnKXxjCa1"
api = lg.Genius(key)

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

def convert(lyrics):
    file = open(lyrics, 'r')
    text = file.read()
    doc = text.split('\n')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(doc)
    ## numpy array of list objects???
    ## indexing last element is giving an issue ------------------|
    sequences = tokenizer.texts_to_sequences(doc) # <---|
    sequences = np.array(sequences)
    input_data = []
    targets = []
    for seq in range(len(sequences)):
        # this is an issue, no index is provided for new line symbol
        if len(sequences[seq]) == 0: continue
        input_data.append(sequences[seq][:-1])
        targets.append(sequences[seq][len(sequences[seq])-1])
    class_size = len(tokenizer.word_index)+1

    input_data = np.array(input_data)
    targets = np.array(targets)
    return input_data, targets, class_size

def search_by_artists(list_of_rappers, filename):
    with open(list_of_rappers) as f:
        file_content = f.readlines()
    file_content = [name.strip() for name in file_content]
    file = open(filename, 'w')
    for rapper in file_content:
        artist = api.search_artist(rapper, max_songs=1, get_full_song_info=False)
        for s in range(len(artist.songs)):
            doc = artist.songs[s].lyrics
            doc = clean_lyrics(doc)
            doc = doc.split('\n')
            for word in doc:
                file.write(word+'\n')
    file.close()


search_by_artists('test.txt', 'out.txt')
X, y, class_size = convert('out.txt')
print("INPUTS: ")
print(np.array(X))
print("TARGETS: ")
print(np.array(y))
