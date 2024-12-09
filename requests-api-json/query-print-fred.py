#!/usr/bin/env python3

import json
import requests

# Query an API, decode the JSON returned.

# My API key (in a URL parameter)
with open('fred.txt', 'r') as file:
    # Read the content of the file
    api_string = file.read().strip()

API_KEY = f"&api_key={api_string}"

# FRED API query URL asking for US population data
FRED_API = "https://api.stlouisfed.org/fred/series/observations?series_id=POP"

# Returned data format
FORMAT = "&file_type=json"

# Time range for data
RANGE = "&observation_start=2021-06-01&observation_end=2022-01-01"

# Complete URL
url = FRED_API + FORMAT + API_KEY + RANGE

# Use the "requests" package to make the query
response = requests.get(url)

# Print raw data (make pretty by cut-n-paste to https://jsonformatter.org/json-pretty-print)
# (see `raw-json-pretty.txt`)
# print("Raw output -> {}".format(response.text))

# Decode the data using the python "json" package
decoded_data = json.loads(response.text)

# Print decoded types
# print(type(decoded_data))
# print(type(decoded_data["observations"]))

# Print some decoded data
print("US Population:")
print("Date\t\tPopulation in millions")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for data_item in decoded_data["observations"]:
    print("{}\t{}".format(data_item["date"], data_item["value"]))
