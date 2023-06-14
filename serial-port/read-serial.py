#!/usr/bin/env python3

import serial

# Raspberry pi rx/tx ports (GPIO14 & GPIO15)
ser = serial.Serial('/dev/serial0')

print('Receiving messages line-by-line ...')

while True:
   line = ser.readline()
   str_line = line.decode('utf8')
   print('  {}'.format(str_line), end='')
   if str_line == 'ABORT\n':
      break

print('Received terminate string')
