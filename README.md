# python-tests

## json-file
* encode data to json and write to file
* read from file and decode json back to python data structures
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

## requests-api-json
* use the requests package to call an api
* decode the results using the inbuilt json package
```
% cd requests-api-json
% /usr/local/bin/pip3 install requests
% ./query-print-fred.py 
US Population:
Date		Population in millions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
2021-06-01	332043.302
2021-07-01	332140.523
...
```

