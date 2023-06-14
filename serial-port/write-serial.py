#!/usr/bin/env python3

import serial

# Write data to serial port

# Messages to be written
messages = ['You may write me down in history', 
    'With your bitter, twisted lies',
    'You may trod me in the very dirt',
    'But still, like dust, I\'ll rise.']

# Raspberry pi rx/tx ports (GPIO14 & GPIO15)
ser = serial.Serial('/dev/serial0')

print('Sending message word-by-word ...')
for message in messages:
    print('  {}'.format(message))
    for word in message.split():
        ser.write(word.encode('utf8'))
    ser.write(b'\n')
    ser.flush()

# Send termination message
termination_string = 'ABORT'
ser.write(termination_string.encode('utf8'))
ser.write(b'\n')
ser.flush()

print('Sent termination message: {}'.format(termination_string))
