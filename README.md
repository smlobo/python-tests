# python-tests

## json-file
```
% cd json-file
% ./data-to-file.py 
Encoded my dictionary to:
{"k1": 1, "k2": "v2", "k3": 3.33}
Wrote my dictionary to json file:  my_dictionary_as_json.json
% ./file-to-data.py 
Read my dictionary from json file:  my_dictionary_as_json.json
{"k1": 1, "k2": "v2", "k3": 3.33}
Decoded to: <class 'dict'>
Read in dictionary value for "k1" :  1
Read in dictionary value for "k2" :  v2
Read in dictionary value for "k3" :  3.33
```