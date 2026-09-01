
# Steps to export playlists from 1 spotify a/c & import into another

## Export from old Spotify a/c

1. On your desktop, login to your OLD spotify a/c
2. Go to https://exportify.net/
3. Click `Get Started`
4. Allow `Exportify` to connect to your Spotify a/c
5. Click `Export All`
6. Click `Log Out`

## Unzip the downloaded zip file

1. `cd; mkdir temp-spotify; cd temp-spotify`
2. `unzip ~/Downloads/spotify_playlist.zip`

## Import into the new Spotify a/c

1. `cd`
2. `git clone git@github.com:smlobo/python-tests.git`
3. `cd python-tests/spotify-import`
4. `python -m pip install -r requirements.txt`
5. Logout of the OLD spotify a/c and login to the NEW a/c in the default browser
6. Go to: https://developer.spotify.com/dashboard/
7. Click `Create App`
8. Name: Transfer; App description: blah; Redirect URIs: http://127.0.0.1:8888/callback
9. Click `Save`
10. `./import_spotify_csv.py ~/temp-spotify/XYZ.csv`


