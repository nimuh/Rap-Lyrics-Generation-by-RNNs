import lyricsgenius as lg
import string
import re
import csv
import pandas
import numpy as np

key = "h7teesaCKXvcUvK8yxFIN1cSBSGchQ6i0MVlwE8YORNByy7N5U19geJqnKXxjCa1"
api = lg.Genius(key)

"""
clean_lyrics is used for removing tags from the lyrics.
"""
def clean_lyrics(doc):
    for word in doc:
        doc = re.sub("'", "", doc)
        doc = re.sub("0|1|2|3|4|5|6|7|8|9", "", doc)
        doc = re.sub("Zero|One|Two|Three|Four|Five|Six|Seven|Eight|Nine",
                     "", doc)
        doc = doc.replace('?', '')
        doc = doc.replace('!', '')
        doc = doc.replace('[', '')
        doc = doc.replace(']', '')
        doc = doc.replace('(', '')
        doc = doc.replace(')', '')
        doc = doc.replace(":", "")
        doc = doc.replace("...", "")
        doc = doc.replace("-", "")
        doc = doc.replace(",", "")
        doc = doc.replace("X", "")
        doc = doc.replace('Verse', '')
        doc = doc.replace('Chorus', '')
        doc = doc.replace('Outro', '')
        doc = doc.replace('Intro', '')
        doc = doc.replace('Hook', '')
        doc = doc.replace('Bridge', '')
        doc = doc.replace('Interlude', '')
    return doc


"""
search_by_artists takes a list of rapper names and searches for
songs by these artists and scrapes the lyrics of songs found. All lyrics
are saved to one file.
"""
def get_lyrics(all_lyrics, out_filename, number_of_songs):
    lyrics = pandas.read_csv(all_lyrics)
    eminem_rap_lyrics = lyrics.loc[lyrics['artist'] == 'Eminem']
    eminem_rap_lyrics = eminem_rap_lyrics.dropna()
    eminem_rap_lyrics = eminem_rap_lyrics['text']
    eminem_rap_lyrics = np.asarray(eminem_rap_lyrics)
    file = open(out_filename, 'w')

    song_nu = 0
    for song in eminem_rap_lyrics:
        if song_nu == number_of_songs:
            break
        current_song = clean_lyrics(song)
        current_song = current_song.split('\n')
        file.write('[startss]' +'\n')
        for word in range(len(current_song)):
            file.write(current_song[word]+'\n')
        file.write('[endss]'+'\n')
        song_nu += 1
        print('Song ', song_nu)
    file.close()


get_lyrics('songdata.csv', 'lyrics.txt', 10)
