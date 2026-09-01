#!/usr/bin/env python3
"""Import the tracks from a Spotify CSV export into one of your playlists.

Set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, and SPOTIPY_REDIRECT_URI before
running this program.  Create the Spotify application at
https://developer.spotify.com/dashboard and add the redirect URI there.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

import spotipy
from spotipy.oauth2 import SpotifyOAuth


SCOPES = "playlist-modify-private playlist-modify-public"
TRACKS_PER_REQUEST = 100


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_tracks(csv_path: Path, client: spotipy.Spotify) -> tuple[list[str], list[str]]:
    """Return track URIs and descriptions that could not be resolved."""
    tracks: list[str] = []
    missing: list[str] = []

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row")

        for row_number, row in enumerate(reader, start=2):
            uri = (row.get("Track URI") or row.get("track_uri") or "").strip()
            if uri.startswith("spotify:track:"):
                tracks.append(uri)
                continue

            title = (row.get("Track Name") or row.get("track_name") or "").strip()
            artist = (row.get("Artist Name(s)") or row.get("artist") or "").strip()
            if not title:
                missing.append(f"row {row_number}: no Track URI or track name")
                continue

            result = client.search(q=f'track:"{title}" artist:"{artist}"', type="track", limit=1)
            items = result["tracks"]["items"]
            if items:
                tracks.append(items[0]["uri"])
            else:
                missing.append(f"row {row_number}: {title} — {artist}")

    return tracks, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a CSV playlist into Spotify.")
    parser.add_argument("csv_file", type=Path, help="CSV export containing Track URI or Track Name/Artist Name(s)")
    parser.add_argument("--name", help="New playlist name (defaults to the CSV filename)")
    parser.add_argument("--public", action="store_true", help="Make the new playlist public")
    parser.add_argument("--description", default="Imported from CSV", help="Playlist description")
    args = parser.parse_args()

    if not args.csv_file.is_file():
        parser.error(f"file not found: {args.csv_file}")
    absent = [key for key in ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REDIRECT_URI") if not os.getenv(key)]
    if absent:
        parser.error("missing environment variables: " + ", ".join(absent))

    client = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES))
    playlist_name = args.name or args.csv_file.stem
    # Spotify removed POST /users/{user_id}/playlists in February 2026.
    # This Spotipy method uses the supported POST /me/playlists endpoint.
    playlist = client.current_user_playlist_create(
        name=playlist_name,
        public=args.public,
        description=args.description,
    )
    tracks, missing = read_tracks(args.csv_file, client)
    for batch in chunks(tracks, TRACKS_PER_REQUEST):
        client.playlist_add_items(playlist["id"], batch)

    print(f"Created {playlist['external_urls']['spotify']} with {len(tracks)} tracks.")
    if missing:
        print(f"Could not find {len(missing)} row(s):", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
