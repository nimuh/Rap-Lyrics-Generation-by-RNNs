import lyricsgenius as lg
import string
import keras
import word2vec
from keras.preprocessing.text import Tokenizer

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
    sequences = tokenizer.texts_to_sequences(doc)
    print(tokenizer.word_index)

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
convert('out.txt')
