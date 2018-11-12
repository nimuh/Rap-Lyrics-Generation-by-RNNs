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
    #with open(list_of_rappers) as f:
        #file_content = f.readlines()
    #file_content = [name.strip() for name in file_content]
    lyrics = pandas.read_csv(all_lyrics)
    rap_lyrics = lyrics.loc[lyrics['genre'] == 'Hip-Hop']
    rap_lyrics = rap_lyrics['lyrics'].dropna()
    rap_lyrics = np.asarray(rap_lyrics)
    file = open(out_filename, 'w')
    #for rapper in file_content:
        #artist = api.search_artist(rapper, max_songs=nu_songs, get_full_song_info=False)
    song_nu = 0
    for song in rap_lyrics:
        song_nu += 1
        if song_nu == number_of_songs:
            break
        #doc = artist.songs[s].lyrics
        current_song = clean_lyrics(song)
        current_song = current_song.split('\n')
        file.write('[startss]' +'\n')
        for word in range(len(current_song)):
            file.write(current_song[word]+'\n')
        file.write('[endss]'+'\n')
        print('Song ', song_nu)
    file.close()


get_lyrics('lyrics.csv', 'lyrics.txt', 100)
