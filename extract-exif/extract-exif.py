#!/usr/bin/env python3.11

import sys
import os
from datetime import datetime

from PIL import Image, ExifTags
from PIL.Image import Exif

import pillow_heif
import exifread
import rawpy
import re

pillow_heif.register_heif_opener()

# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------


def parse_iso6709(value):
    match = re.match(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', value)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def convert_to_degrees(value):
    d, m, s = value
    return float(d) + float(m) / 60 + float(s) / 3600


def parse_exif_date(value):
    try:
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return value


def extract_gps_from_dict(gps):
    try:
        lat = convert_to_degrees(gps["GPSLatitude"])
        if gps["GPSLatitudeRef"] != "N":
            lat = -lat

        lon = convert_to_degrees(gps["GPSLongitude"])
        if gps["GPSLongitudeRef"] != "E":
            lon = -lon

        return lat, lon
    except Exception:
        return None


# ------------------------------------------------------------
# JPEG / TIFF / HEIC via Pillow
# ------------------------------------------------------------

def extract_with_pillow(path):
    with Image.open(path) as img:
        exif = img._getexif() or {}

    decoded = {}
    for tag, value in exif.items():
        decoded[ExifTags.TAGS.get(tag, tag)] = value

    gps = {}
    if "GPSInfo" in decoded:
        for k, v in decoded["GPSInfo"].items():
            gps[ExifTags.GPSTAGS.get(k, k)] = v

    coords = extract_gps_from_dict(gps) if gps else None

    dates = {}
    for k in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        if k in decoded:
            dates[k] = parse_exif_date(decoded[k])

    return coords, dates


# ------------------------------------------------------------
# HEIC support
# ------------------------------------------------------------

# def extract_heic(path):
#     heif = pillow_heif.open_heif(path)
#     img = Image.frombytes(
#         heif.mode, heif.size, heif.data, "raw", heif.mode, heif.stride
#     )
#     img.info["exif"] = heif.exif
#     return extract_with_pillow(img)

def extract_heic(path):
    with Image.open(path) as img:
        exif_bytes = img.info.get("exif")
        if not exif_bytes:
            return None, {}

        exif = Exif()
        exif.load(exif_bytes)

    decoded = {}
    for tag_id, value in exif.items():
        print(f'Exif items: {tag_id} -> {value}')
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        decoded[tag] = value
        print(f'  decoded: {tag} -> {value}')

    gps = {}
    if "GPSInfo" in decoded:
        print(f'Found GPSInfo')
        # for k, v in decoded["GPSInfo"].items():
        for k, v in decoded["GPSInfo"]:
            print(f'GPSInfo: {k} -> {v}')
            gps[ExifTags.GPSTAGS.get(k, k)] = v
    # gps = {}

    # gps_ifd = exif.get_ifd(ExifTags.IFD.GPS)
    # if gps_ifd:
    #     for tag_id, value in gps_ifd.items():
    #         tag_name = ExifTags.GPSTAGS.get(tag_id, tag_id)
    #         gps[tag_name] = value

    # # Fallback: Apple ISO 6709 location
    # iso_loc = exif.get(0x8833)  # LocationISO6709 tag
    # if iso_loc:
    #     print(f'Found iso_loc: {iso_loc}')
    #     coords = parse_iso6709(str(iso_loc))
    #     if coords:
    #         return coords, dates

    coords = extract_gps_from_dict(gps) if gps else None

    dates = {}
    for k in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        if k in decoded:
            dates[k] = parse_exif_date(decoded[k])

    return coords, dates


# ------------------------------------------------------------
# DNG (RAW) via exifread
# ------------------------------------------------------------

def extract_dng(path):
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    def ratio_to_float(r):
        return float(r.num) / float(r.den)

    gps = {}
    try:
        gps["GPSLatitude"] = tuple(ratio_to_float(r) for r in tags["GPS GPSLatitude"].values)
        gps["GPSLatitudeRef"] = str(tags["GPS GPSLatitudeRef"])
        gps["GPSLongitude"] = tuple(ratio_to_float(r) for r in tags["GPS GPSLongitude"].values)
        gps["GPSLongitudeRef"] = str(tags["GPS GPSLongitudeRef"])
    except Exception:
        gps = None

    coords = extract_gps_from_dict(gps) if gps else None

    dates = {}
    for key in ("EXIF DateTimeOriginal", "Image DateTime"):
        if key in tags:
            dates[key] = parse_exif_date(tags[key])

    return coords, dates


# ------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------

def extract_metadata(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in (".jpg", ".jpeg", ".tif", ".tiff"):
        return extract_with_pillow(path)

    if ext == ".heic":
        return extract_heic(path)

    if ext == ".dng":
        return extract_dng(path)

    raise ValueError(f"Unsupported format: {ext}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_exif.py <image1> [image2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\nFile: {path}")

        if not os.path.isfile(path):
            print("  Not a file")
            continue

        try:
            coords, dates = extract_metadata(path)

            if coords:
                # print(f"  GPS: latitude={coords[0]:.6f}, longitude={coords[1]:.6f}")
                print(f"  GPS:\n{coords[0]:.6f}, {coords[1]:.6f}")
            else:
                print("  GPS: not available")

            if dates:
                for k, v in dates.items():
                    print(f"  {k}: {v}")
            else:
                print("  Date: not available")

        except Exception as e:
            print(f"  Error: {e}")
