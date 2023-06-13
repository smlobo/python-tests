#!/usr/bin/env python3

import json

# Create a dictionary
my_dictionary = {'k1': 1, 'k2': "v2", 'k3': 3.33}

# Encode for writing to file
encoded_as_json = json.dumps(my_dictionary)
print("Encoded my dictionary to:")
print(encoded_as_json)

# Write to file
file_name = "my_dictionary_as_json.json"
file_handler = open(file_name, "w")
file_handler.write(encoded_as_json)
file_handler.close()
print("Wrote my dictionary to json file: ", file_name)
