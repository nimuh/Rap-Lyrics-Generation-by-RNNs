import string
import re
import csv
import pandas
import numpy as np


def clean_lyrics(doc):
    """
    Used for cleaning lyrics.

    # Arguments:
        - doc: The lyrics document to be cleaned.
    """

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
    doc = doc.replace(";", "")
    doc = doc.replace("#", "")
    doc = doc.replace("/", "")
    doc = doc.replace("&", "")
    doc = doc.replace("*", "")
    doc = doc.replace("$", "")
    doc = doc.replace("~", "")
    doc = doc.replace("+", "")
    doc = doc.replace("...", "")
    doc = doc.replace("-", "")
    doc = doc.replace(",", "")
    doc = doc.replace("X", "")
    doc = doc.replace("Marshall", "")
    doc = doc.replace("Eminem", "")
    doc = doc.replace("Dr. Dre", "")
    doc = doc.replace("Joe Beast", "")
    doc = doc.replace('Verse', '')
    doc = doc.replace('Chorus x', '')
    doc = doc.replace("Chorus", '')
    doc = doc.replace('Intro', '')
    doc = doc.replace('Hook', '')
    doc = doc.replace('"', '')
    doc = doc.replace("'", '')
    doc = doc.replace(".", " ")
    doc = doc.lower()
    return doc


def get_lyrics(all_lyrics, out_filename, number_of_songs):

    """
    Uses lyrics data set to create lyrics document to generated input and output
    sequences.

    Artists:
        - Eminem
        - Rihanna
        - Nicki Minaj
        - Kanye West
    # Arguments:
        - all_lyrics: A CSV file containing lyrics from different artists.
        - out_filename: The file that the chosen lyrics will be written to.
        - number_of_songs: Number of Eminem songs to use. This was originally
                           used to limit the size of the data I was working
                           with. 70 is the maximum number of Eminem songs in
                           all_lyrics.
    """
    lyrics = pandas.read_csv(all_lyrics)
    eminem_rap_lyrics = lyrics.loc[lyrics['artist'] == 'Eminem']
    eminem_rap_lyrics = eminem_rap_lyrics.dropna()
    eminem_rap_lyrics = eminem_rap_lyrics['text']
    eminem_rap_lyrics = np.asarray(eminem_rap_lyrics)
    nm_lyrics = lyrics.loc[lyrics['artist'] == 'Nicki Minaj']
    nm_lyrics = nm_lyrics.dropna()
    nm_lyrics = nm_lyrics['text']
    nm_lyrics = np.asarray(nm_lyrics)

    rha_lyrics = lyrics.loc[lyrics['artist'] == 'Rihanna']
    rha_lyrics = rha_lyrics.dropna()
    rha_lyrics = rha_lyrics['text']
    rha_lyrics = np.asarray(rha_lyrics)


    file = open(out_filename, 'w')

    for song in rha_lyrics:
	    current_song = clean_lyrics(song)
	    current_song = current_song.split('\n')
	    for word in range(len(current_song)):
	        file.write(current_song[word]+'\n')

    for song in nm_lyrics:
	    current_song = clean_lyrics(song)
	    current_song = current_song.split('\n')
	    for word in range(len(current_song)):
	        file.write(current_song[word]+'\n')

    song_nu = 0
    for song in eminem_rap_lyrics:
        if song_nu == number_of_songs:
            break
        current_song = clean_lyrics(song)
        current_song = current_song.split('\n')
        for word in range(len(current_song)):
            file.write(current_song[word]+'\n')
        song_nu += 1

    with open("kanye_verses.txt") as f:
        kanye_verses = f.readlines()

    kanye_songs = [ver.strip() for ver in kanye_verses]
    for verse in kanye_songs:
	    current_verse = clean_lyrics(verse)
	    file.write(current_verse+'\n')

    file.close()

"""
########################## GET LYRICS #########################################
"""
get_lyrics('songdata.csv', 'lyrics.txt', 70)
