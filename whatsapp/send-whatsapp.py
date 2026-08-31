#!/usr/bin/env python3

import json
import requests

# POST to meta API for sending a whatsapp message

# Whatsapp phone number id & access token
with open('whatsapp.json', 'r') as file:
    # Read & decode the content of the file
    account_data = json.load(file)

# print(f'Token: {account_data["access_token"]}')

# Whatsapp URL for sending messages
phone_id = account_data["phone_number_id"]
whatsapp_url = f'https://graph.facebook.com/v21.0/{phone_id}/messages'
print(f'{whatsapp_url}')

# HTTP headers
headers = {}
headers["Authorization"] = f'Bearer {account_data["access_token"]}'
headers["Content-Type"] = 'application/json'

# Payload data
post_data = {}
post_data["messaging_product"] = 'whatsapp'
post_data["to"] = '19199616206'
post_data["type"] = 'text'
post_data["text"] = {"body": "test from python"}

# Send
response = requests.post(whatsapp_url, json=post_data, headers=headers)

# Print the response
print(response.status_code)
print(response.json())
