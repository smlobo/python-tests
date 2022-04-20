#!/usr/local/bin/python3

import json

# Read from file
file_name = "my_dictionary_as_json.json"
file_handler = open(file_name, "r")
raw_data = file_handler.read()
file_handler.close()
print("Read my dictionary from json file: ", file_name)
print(raw_data)

# Decode from JSON to my dictionary
decoded_data = json.loads(raw_data)
print("Decoded to: {}".format(type(decoded_data)))

# Print keys to test
print("Read in dictionary value for \"k1\" : ", decoded_data["k1"])
print("Read in dictionary value for \"k2\" : ", decoded_data["k2"])
print("Read in dictionary value for \"k3\" : ", decoded_data["k3"])
