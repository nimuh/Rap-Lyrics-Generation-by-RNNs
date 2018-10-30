import lyricsgenius as lg
import string

key = "h7teesaCKXvcUvK8yxFIN1cSBSGchQ6i0MVlwE8YORNByy7N5U19geJqnKXxjCa1"
api = lg.Genius(key)

"""
clean_lyrics is used for removing tags from the lyrics.
"""
def clean_lyrics(doc):
    for word in doc:
        doc.replace('?', '')
        doc.replace('!', '')
        doc = doc.replace('[', '')
        doc = doc.replace(']', '')
        doc = doc.replace('(', '')
        doc = doc.replace(')', '')
        doc = doc.replace('1', '')
        doc = doc.replace('2', '')
        doc = doc.replace('3', '')
        doc = doc.replace('Verse', '')
        doc = doc.replace('Chorus', '')
        doc = doc.replace('Outro', '')
        doc = doc.replace('Intro', '')
        doc = doc.replace('Hook', '')
        doc.replace('Bridge', '')
        doc = doc.replace('Interlude', '')
    return doc


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
            for word in range(len(doc)):
                file.write(doc[word]+ ' [END]' + '\n')
    file.close()


search_by_artists('test.txt', 'lyrics.txt', 1)
